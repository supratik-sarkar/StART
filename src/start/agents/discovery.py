"""Data-first discovery agents.

These are the entry point of the model-review operating layer: before any
model is chosen, the framework inspects the dataset, surfaces candidate
targets, and infers the modeling task — each step emitting an evidence record
so the review is grounded from the very first interaction.

    DatasetDiscoveryAgent  -> schema, types, candidate targets/timestamps/
                              entities, text + image-path columns, missingness
    TargetDiscoveryAgent   -> ranked target candidates (user confirms; no
                              training without explicit selection)
    TaskInferenceAgent     -> binary / multiclass / multilabel / regression /
                              forecasting / ranking / recommendation / anomaly

The LLM is never involved here in deterministic mode. In LLM mode the
prompt-guided intake (see start.agents.intent) may *propose* targets/task, but
only the deterministic, evidence-backed result drives execution unless the
user overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from start.core.schemas import Status, TestResult

# Task taxonomy (item: task inference)
TASK_TYPES = (
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "forecasting",
    "ranking",
    "recommendation",
    "anomaly_detection",
)

_ID_HINTS = ("id", "key", "uuid", "guid", "identifier")
_TIME_HINTS = ("date", "time", "timestamp", "ts", "datetime", "period")
_IMAGE_HINTS = ("path", "image", "img", "file", "filename", "filepath")


def _tokens(name: str) -> list[str]:
    """Split a column name into lowercased tokens on separators and camelCase."""
    import re

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [t for t in re.split(r"[\s_\-./]+", spaced.lower()) if t]


def _matches_hint(name: str, hints: tuple[str, ...]) -> bool:
    """Token-aware hint match: a hint matches only as a whole token (or, for
    longer hints >= 4 chars, as a token prefix/suffix). Avoids false positives
    like 'ts' inside 'points' or 'id' inside 'width'."""
    toks = set(_tokens(name))
    for h in hints:
        if h in toks:
            return True
        if len(h) >= 4 and any(t.startswith(h) or t.endswith(h) for t in toks):
            return True
    return False


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str  # numeric | categorical | text | datetime | image_path | identifier | boolean
    n_unique: int
    missing_pct: float


@dataclass
class DiscoveryProfile:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    candidate_targets: list[str]
    timestamp_columns: list[str]
    entity_columns: list[str]
    text_columns: list[str]
    image_path_columns: list[str]
    high_missing_columns: list[str]
    notes: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def summary(self) -> str:
        return (
            f"{self.n_rows} rows x {self.n_columns} columns; "
            f"{len(self.candidate_targets)} candidate target(s), "
            f"{len(self.text_columns)} text, {len(self.image_path_columns)} image-path, "
            f"{len(self.timestamp_columns)} timestamp, {len(self.entity_columns)} entity column(s)."
        )


def _looks_like_image_path(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
    hits = sample.str.lower().str.endswith(exts)
    return bool(hits.mean() > 0.5)


def _column_role(name: str, series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        if _matches_hint(name, _ID_HINTS) and nunique > 0.9 * len(series):
            return "identifier"
        return "numeric"
    # object / string dtypes
    if _looks_like_image_path(series):
        return "image_path"
    non_null = series.dropna()
    avg_len = non_null.astype(str).str.len().mean() if not non_null.empty else 0
    if avg_len > 30:
        return "text"
    return "categorical"


class DatasetDiscoveryAgent:
    """Inspect a DataFrame and produce a structured discovery profile."""

    def discover(self, df: pd.DataFrame) -> DiscoveryProfile:
        columns: list[ColumnProfile] = []
        timestamps, entities, texts, images, high_missing = [], [], [], [], []
        for name in df.columns:
            series = df[name]
            role = _column_role(name, series)
            missing = float(series.isna().mean() * 100)
            columns.append(
                ColumnProfile(
                    name=name,
                    dtype=str(series.dtype),
                    role=role,
                    n_unique=int(series.nunique(dropna=True)),
                    missing_pct=round(missing, 4),
                )
            )
            if role == "datetime" or _matches_hint(name, _TIME_HINTS):
                timestamps.append(name)
            if role == "identifier" or _matches_hint(name, _ID_HINTS):
                entities.append(name)
            if role == "text":
                texts.append(name)
            if role == "image_path":
                images.append(name)
            if missing > 30.0:
                high_missing.append(name)

        candidates = self._candidate_targets(df, columns, set(entities) | set(images))
        return DiscoveryProfile(
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=columns,
            candidate_targets=candidates,
            timestamp_columns=timestamps,
            entity_columns=entities,
            text_columns=texts,
            image_path_columns=images,
            high_missing_columns=high_missing,
        )

    def _candidate_targets(
        self, df: pd.DataFrame, columns: list[ColumnProfile], exclude: set[str]
    ) -> list[str]:
        _PROTECTED_OR_AUX_HINTS = {
            "foreign_worker",
            "telephone",
            "phone",
            "liable_people",
            "people_liable",
            "personal_status",
            "sex",
            "gender",
            "race",
            "ethnicity",
            "age",
            "marital_status",
            "nationality",
            "religion",
            "disability",
        }
        scored: list[tuple[float, str]] = []
        last_col = df.columns[-1] if len(df.columns) > 0 else ""
        for col in columns:
            if col.name in exclude or col.role in {"datetime", "image_path", "identifier", "text"}:
                continue
            lname = col.name.lower()
            if any(k in lname for k in _PROTECTED_OR_AUX_HINTS):
                continue
            score = 0.0
            if lname in {
                "target",
                "label",
                "y",
                "class",
                "outcome",
                "default",
                "churn",
                "fraud",
                "attrition",
                "status",
                "credit_risk",
                "response",
                "bad",
            }:
                score += 5
            elif any(
                k in lname
                for k in (
                    "target",
                    "label",
                    "churn",
                    "default",
                    "fraud",
                    "attrition",
                    "credit_risk",
                    "response",
                    "bad",
                )
            ):
                score += 3
            if col.name == last_col and (
                col.role in {"boolean", "categorical"} or (col.role == "numeric" and col.n_unique <= 5)
            ):
                score += 2.5
            if col.missing_pct > 5:
                score -= 1.5
            if score >= 2.5:
                scored.append((score, col.name))
        scored.sort(reverse=True)
        return [name for _, name in scored]

    def to_evidence(self, profile: DiscoveryProfile) -> TestResult:
        result = TestResult(
            test_id="discovery.dataset_profile",
            test_name="Dataset discovery profile",
            metrics={
                "n_rows": profile.n_rows,
                "n_columns": profile.n_columns,
                "n_candidate_targets": len(profile.candidate_targets),
                "candidate_targets": ", ".join(profile.candidate_targets[:8]),
                "text_columns": ", ".join(profile.text_columns[:8]),
                "image_path_columns": ", ".join(profile.image_path_columns[:8]),
                "timestamp_columns": ", ".join(profile.timestamp_columns[:8]),
                "entity_columns": ", ".join(profile.entity_columns[:8]),
                "high_missing_columns": ", ".join(profile.high_missing_columns[:8]),
            },
            interpretation=profile.summary(),
            limitations=[
                "Roles are inferred heuristically; confirm targets and entities before training.",
            ],
        )
        return result.apply_thresholds()

    def analyze_categorical_density(self, df: pd.DataFrame) -> tuple[str, float, str]:
        """Scans data schemas for high-cardinality blocks to recommend CatBoost or Distributed Random Forest."""
        total_features = len(df.columns)
        categorical_features_count = 0
        high_cardinality_detected = False

        for col in df.columns:
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
                categorical_features_count += 1
                unique_count = df[col].nunique()
                if unique_count > 50:
                    high_cardinality_detected = True

        categorical_ratio = categorical_features_count / max(total_features, 1)

        # Core heuristic decision boundary logic
        if categorical_ratio > 0.20 or high_cardinality_detected:
            model_rec = "catboost"
            reasoning = f"High categorical density ({categorical_ratio * 100:.1f}%) or cardinality detected. Explicitly recommending CatBoost over standard trees to preserve structural feature interactions natively."
        else:
            model_rec = "distributed_random_forest"
            reasoning = "Dense numerical feature distribution with bounded cardinality. Recommending Distributed Random Forest parameters to maximize split variance across parallel trees."

        return model_rec, float(categorical_ratio), reasoning


@dataclass
class TargetRecommendation:
    candidates: list[str]
    selected: str | list[str] | None = None
    multi_output: bool = False
    note: str = ""


class TargetDiscoveryAgent:
    """Surface ranked target candidates; the user must explicitly select. No
    training proceeds on an unconfirmed target."""

    def recommend(
        self, profile: DiscoveryProfile, user_target: str | list[str] | None = None
    ) -> TargetRecommendation:
        if user_target is not None:
            multi = isinstance(user_target, (list, tuple)) and len(user_target) > 1
            return TargetRecommendation(
                candidates=profile.candidate_targets,
                selected=list(user_target) if multi else user_target,
                multi_output=multi,
                note="User-confirmed target.",
            )
        if not profile.candidate_targets:
            return TargetRecommendation(
                candidates=[],
                note="No clear target candidate found; the user must specify one explicitly.",
            )
        return TargetRecommendation(
            candidates=profile.candidate_targets,
            note=(
                f"Top candidate is '{profile.candidate_targets[0]}'. "
                "Confirm explicitly before training (no automatic target selection)."
            ),
        )

    def to_evidence(self, rec: TargetRecommendation) -> TestResult:
        status = Status.PASS if rec.selected is not None else Status.WARN
        return TestResult(
            test_id="discovery.target_selection",
            test_name="Target selection",
            status=status,
            metrics={
                "candidates": ", ".join(rec.candidates[:8]),
                "selected": str(rec.selected) if rec.selected is not None else "(unconfirmed)",
                "multi_output": str(rec.multi_output),
            },
            interpretation=rec.note,
            limitations=["Training requires an explicitly confirmed target."],
        )


@dataclass
class TaskInference:
    task_type: str
    target_type: str
    n_classes: int | None = None
    overridden: bool = False
    note: str = ""


class TaskInferenceAgent:
    """Infer the modeling task from the target column(s); user-overridable."""

    def infer(
        self,
        df: pd.DataFrame,
        target: str | list[str],
        *,
        override: str | None = None,
        has_timestamp: bool = False,
    ) -> TaskInference:
        if override:
            if override not in TASK_TYPES:
                raise ValueError(f"Unknown task '{override}'. Known: {TASK_TYPES}")
            return TaskInference(
                task_type=override,
                target_type="user",
                overridden=True,
                note=f"Task overridden by user to {override}.",
            )

        if isinstance(target, (list, tuple)) and len(target) > 1:
            # multi-output: multilabel if all binary, else multi-output regression-ish
            all_binary = all(df[t].dropna().nunique() == 2 for t in target)
            task = "multilabel_classification" if all_binary else "regression"
            return TaskInference(
                task_type=task,
                target_type="multi_output",
                n_classes=len(target),
                note=f"{len(target)} target columns -> {task}.",
            )

        col = target[0] if isinstance(target, (list, tuple)) else target
        series = df[col].dropna()
        nunique = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_integer = pd.api.types.is_integer_dtype(series)

        if nunique == 2:
            task, ttype = "binary_classification", "binary"
        elif nunique <= 20 and (not is_numeric or is_integer):
            task, ttype = "multiclass_classification", "multiclass"
        elif is_numeric:
            task = "forecasting" if has_timestamp else "regression"
            ttype = "continuous"
        else:
            task, ttype = "multiclass_classification", "multiclass"

        return TaskInference(
            task_type=task,
            target_type=ttype,
            n_classes=nunique if "classification" in task else None,
            note=f"Inferred {task} from target '{col}' ({nunique} unique values).",
        )

    def to_evidence(self, inference: TaskInference) -> TestResult:
        result = TestResult(
            test_id="discovery.task_inference",
            test_name="Task inference",
            metrics={
                "task_type": inference.task_type,
                "target_type": inference.target_type,
                "n_classes": inference.n_classes if inference.n_classes is not None else 0,
                "overridden": str(inference.overridden),
            },
            interpretation=inference.note,
            limitations=["Inference is heuristic; the user may override the task type."],
        )
        return result.apply_thresholds()


def run_discovery(
    df: pd.DataFrame,
    *,
    user_target: str | list[str] | None = None,
    task_override: str | None = None,
) -> tuple[DiscoveryProfile, TargetRecommendation, TaskInference, list[TestResult]]:
    """Convenience: run all three discovery agents and collect their evidence."""
    discovery = DatasetDiscoveryAgent()
    profile = discovery.discover(df)
    target_agent = TargetDiscoveryAgent()
    target_rec = target_agent.recommend(profile, user_target)
    evidence = [discovery.to_evidence(profile), target_agent.to_evidence(target_rec)]

    inference: TaskInference | None = None
    chosen = target_rec.selected or (profile.candidate_targets[0] if profile.candidate_targets else None)
    if chosen is not None:
        inference = TaskInferenceAgent().infer(
            df,
            chosen,
            override=task_override,
            has_timestamp=bool(profile.timestamp_columns),
        )
        evidence.append(TaskInferenceAgent().to_evidence(inference))
    return profile, target_rec, inference, evidence
