"""Execution Policies for StART Scientific Certification: Deterministic vs. GPT-4.1.

Strict Invariant: GPT41_NUMERIC_AUTHORITY == 0.
OpenAI gpt-4.1 is used solely for bounded architectural and hyperparameter planning.
All numeric evaluation, metric calculations, and scientific invariants run through
the exact same deterministic StART scientific engines.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from start.certification.spec import ProviderTraceRecord

logger = logging.getLogger("start.certification.policies")

GPT41_MODEL = "gpt-4.1"
GPT41_NUMERIC_AUTHORITY = 0


@dataclass
class PolicyPlan:
    """Structured plan emitted by either Deterministic or GPT-4.1 policy."""

    policy: Literal["deterministic", "gpt41"]
    domain: str
    model: str
    preprocessing: dict[str, Any]
    split: dict[str, Any]
    tuning_strategy: str
    trial_budget: int
    primary_metric: str
    secondary_metrics: list[str]
    xai_methods: list[str]
    sensitivity_mode: Literal["one_at_a_time", "parallel_basket", "both"]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    trace_record: ProviderTraceRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.trace_record:
            d["trace_record"] = self.trace_record.to_dict()
        return d


class DeterministicPolicyRunner:
    """Predeclared engineering baseline policy.

    Emits standardized, reproducible configurations for each domain.
    """

    def plan(self, domain: str, dataset_meta: dict[str, Any], model_override: str | None = None) -> PolicyPlan:
        dom = domain.lower().strip()
        if dom == "predictive":
            model = model_override or "xgboost"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={
                    "imputation": "median",
                    "scaling": "standard",
                    "outlier_clipping": True,
                    "clip_std": 3.0,
                },
                split={"strategy": "stratified", "test_size": 0.20},
                tuning_strategy="bounded_random_search",
                trial_budget=5,
                primary_metric="roc_auc",
                secondary_metrics=["pr_auc", "f1", "precision", "recall", "brier"],
                xai_methods=["shap_tree_explainer", "permutation_importance", "native_importance"],
                sensitivity_mode="both",
                hyperparameters={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
                rationale="Standard production configuration for tabular binary classification.",
            )

        elif dom in ("fraud", "fraud_aml"):
            model = model_override or "random_forest"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={
                    "imputation": "median",
                    "scaling": "robust",
                    "outlier_clipping": False,
                    "class_weight": "balanced",
                },
                split={"strategy": "stratified", "test_size": 0.20},
                tuning_strategy="bounded_random_search",
                trial_budget=5,
                primary_metric="pr_auc",
                secondary_metrics=["roc_auc", "f1", "recall_at_fdr", "matthews_corr"],
                xai_methods=["permutation_importance", "native_importance"],
                sensitivity_mode="both",
                hyperparameters={"n_estimators": 100, "class_weight": "balanced", "max_depth": 8},
                rationale="Cost-sensitive imbalanced fraud detection with PR-AUC primary objective.",
            )

        elif dom in ("deep_learning", "tabular_dl"):
            model = model_override or "mlp"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={
                    "imputation": "mean",
                    "scaling": "standard",
                    "outlier_clipping": True,
                },
                split={"strategy": "stratified", "test_size": 0.20},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="roc_auc",
                secondary_metrics=["pr_auc", "f1", "brier", "latency_ms"],
                xai_methods=["permutation_importance"],
                sensitivity_mode="one_at_a_time",
                hyperparameters={"hidden_dims": [64, 32], "learning_rate": 0.001, "epochs": 10},
                rationale="PyTorch Tabular MLP architecture with cross-entropy loss.",
            )

        elif dom == "recommender":
            model = model_override or "ffm"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={"field_aware": True, "min_user_interactions": 3},
                split={"strategy": "leave_one_out", "test_size": 0.20},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="ndcg@10",
                secondary_metrics=["map@10", "mrr", "hit_rate@10"],
                xai_methods=["field_interaction_weights"],
                sensitivity_mode="one_at_a_time",
                hyperparameters={"latent_dim": 4, "learning_rate": 0.02, "epochs": 10},
                rationale="Field-Aware Factorization Machine (FFM) with field-to-field interaction matrix.",
            )

        elif dom == "portfolio":
            model = model_override or "hrp"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={"returns_frequency": "daily", "annualization_factor": 252},
                split={"strategy": "in_sample_out_sample", "test_size": 0.30},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="diversification_ratio",
                secondary_metrics=["volatility", "sharpe", "sortino", "max_drawdown"],
                xai_methods=["risk_contribution_decomposition"],
                sensitivity_mode="parallel_basket",
                hyperparameters={"linkage_method": "single", "distance_metric": "correlation"},
                rationale="Hierarchical Risk Parity (HRP) via quasi-diagonalization and recursive bisection.",
            )

        elif dom == "market_risk":
            model = model_override or "cornish_fisher"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={"sign_convention": "loss_positive", "confidence_level": 0.99},
                split={"strategy": "sliding_window", "window_size": 250},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="kupiec_pof_pvalue",
                secondary_metrics=["christoffersen_pvalue", "var_99", "es_99", "exception_count"],
                xai_methods=["component_var_attribution"],
                sensitivity_mode="one_at_a_time",
                hyperparameters={"alpha": 0.99, "horizon_days": 10},
                rationale="Cornish-Fisher expansion accounting for skewness and kurtosis in tail quantiles.",
            )

        elif dom in ("scenario", "traded_risk"):
            model = model_override or "gate6_tail_shock"
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model=model,
                preprocessing={"repricing_method": "delta_gamma"},
                split={"strategy": "full_portfolio"},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="repricing_consistency",
                secondary_metrics=["max_tail_loss", "monotonicity_pass", "factor_shock_fidelity"],
                xai_methods=["scenario_factor_attribution"],
                sensitivity_mode="parallel_basket",
                hyperparameters={"shocks": {"equity": -0.20, "credit_spread": 0.50, "rates": -0.01}},
                rationale="Gate-6/6A regulatory asset tail stress shock specification.",
            )

        else:
            return PolicyPlan(
                policy="deterministic",
                domain=domain,
                model="random_forest",
                preprocessing={"imputation": "median", "scaling": "standard"},
                split={"strategy": "stratified", "test_size": 0.20},
                tuning_strategy="deterministic",
                trial_budget=1,
                primary_metric="roc_auc",
                secondary_metrics=["f1"],
                xai_methods=["permutation_importance"],
                sensitivity_mode="one_at_a_time",
                hyperparameters={"n_estimators": 100},
                rationale="Fallback deterministic plan.",
            )


class GPT41PolicyRunner:
    """Bounded Generative AI Planning Policy using OpenAI gpt-4.1.

    STRICT CONSTRAINTS:
    - Model: OpenAI gpt-4.1 only (GPT41_MODEL).
    - Numeric Authority: Exactly 0. Never computes metrics or numbers.
    - Max Calls: 1 primary planning call + at most 1 bounded schema-correction call.
    - All proposed plans run on the exact same deterministic StART scientific engines.
    """

    def __init__(self) -> None:
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI

            from start.providers.keys import ensure_provider_key

            status = ensure_provider_key("openai")
            logger.info("GPT41PolicyRunner: OpenAI key status: %s", status)
            self.client = OpenAI()
        except Exception as exc:
            logger.warning("GPT41PolicyRunner failed to initialize OpenAI client: %s", exc)
            self.client = None

    def plan(
        self,
        domain: str,
        experiment_id: str,
        dataset_meta: dict[str, Any],
        allowed_models: list[str],
        allowed_metrics: list[str],
    ) -> PolicyPlan:
        """Call OpenAI gpt-4.1 to generate an architectural plan."""
        call_id = f"gpt41_plan_{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()

        system_prompt = (
            "You are an expert quantitative machine learning architect for the StART risk and modeling platform. "
            "Your role is STRICTLY BOUNDED to algorithmic planning and configuration. "
            "CRITICAL INVARIANT: You have ZERO NUMERIC AUTHORITY. You do NOT compute metrics, losses, scores, or predictions. "
            "You only select valid models, preprocessing strategies, hyperparameter search spaces, and metrics "
            "from the declared capabilities.\n\n"
            "Respond ONLY with a valid JSON object adhering to this schema:\n"
            "{\n"
            '  "selected_model": "<string from allowed_models>",\n'
            '  "preprocessing": {\n'
            '    "imputation": "median" | "mean" | "zero",\n'
            '    "scaling": "standard" | "robust" | "minmax" | "none",\n'
            '    "outlier_clipping": true | false\n'
            "  },\n"
            '  "split": {\n'
            '    "strategy": "stratified" | "random",\n'
            '    "test_size": 0.20\n'
            "  },\n"
            '  "tuning_strategy": "bounded_random_search" | "optuna" | "deterministic",\n'
            '  "trial_budget": 5,\n'
            '  "primary_metric": "<string from allowed_metrics>",\n'
            '  "secondary_metrics": ["<strings from allowed_metrics>"],\n'
            '  "xai_methods": ["shap_tree_explainer", "permutation_importance", "native_importance"],\n'
            '  "sensitivity_mode": "one_at_a_time" | "parallel_basket" | "both",\n'
            '  "hyperparameters": {"n_estimators": 150, "max_depth": 6},\n'
            '  "architectural_rationale": "<concise explanation>"\n'
            "}"
        )

        user_prompt = (
            f"Domain: {domain}\n"
            f"Dataset Metadata: {json.dumps(dataset_meta)}\n"
            f"Allowed Models: {json.dumps(allowed_models)}\n"
            f"Allowed Metrics: {json.dumps(allowed_metrics)}\n\n"
            "Design the optimal scientific model training plan for this dataset and domain."
        )

        trace = ProviderTraceRecord(
            call_id=call_id,
            timestamp=time.time(),
            provider="OpenAI",
            model=GPT41_MODEL,
            experiment_id=experiment_id,
            prompt=user_prompt,
            response_raw="",
            response_parsed={},
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=0.0,
            schema_valid=False,
            correction_call_required=False,
        )

        if not self.client:
            logger.warning("OpenAI client not available, returning deterministic baseline plan.")
            det_plan = DeterministicPolicyRunner().plan(domain, dataset_meta)
            det_plan.policy = "gpt41"
            det_plan.rationale = "OpenAI client unavailable; defaulted through deterministic engine."
            det_plan.trace_record = trace
            return det_plan

        try:
            res = self.client.chat.completions.create(
                model=GPT41_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw_content = res.choices[0].message.content or "{}"
            prompt_tokens = res.usage.prompt_tokens if res.usage else 0
            completion_tokens = res.usage.completion_tokens if res.usage else 0
            latency = time.perf_counter() - t0

            trace.response_raw = raw_content
            trace.prompt_tokens = prompt_tokens
            trace.completion_tokens = completion_tokens
            trace.latency_seconds = latency

            parsed = json.loads(raw_content)
            valid = self._validate_plan(parsed, allowed_models, allowed_metrics)
            trace.schema_valid = valid
            trace.response_parsed = parsed

            if not valid:
                logger.warning("Primary GPT-4.1 response failed schema validation. Triggering bounded correction call.")
                trace.correction_call_required = True
                t_corr0 = time.perf_counter()
                corr_res = self.client.chat.completions.create(
                    model=GPT41_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": raw_content},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response had schema validation errors. "
                                f"Ensure 'selected_model' is in {allowed_models} and 'primary_metric' is in {allowed_metrics}. "
                                "Output corrected JSON only."
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
                corr_raw = corr_res.choices[0].message.content or "{}"
                trace.prompt_tokens += corr_res.usage.prompt_tokens if corr_res.usage else 0
                trace.completion_tokens += corr_res.usage.completion_tokens if corr_res.usage else 0
                trace.latency_seconds += time.perf_counter() - t_corr0
                trace.response_raw = corr_raw
                parsed = json.loads(corr_raw)
                trace.schema_valid = self._validate_plan(parsed, allowed_models, allowed_metrics)
                trace.response_parsed = parsed

            selected_model = parsed.get("selected_model", allowed_models[0])
            if selected_model not in allowed_models:
                selected_model = allowed_models[0]

            primary_metric = parsed.get("primary_metric", allowed_metrics[0])
            if primary_metric not in allowed_metrics:
                primary_metric = allowed_metrics[0]

            plan = PolicyPlan(
                policy="gpt41",
                domain=domain,
                model=selected_model,
                preprocessing=parsed.get("preprocessing", {"imputation": "median", "scaling": "standard"}),
                split=parsed.get("split", {"strategy": "stratified", "test_size": 0.20}),
                tuning_strategy=parsed.get("tuning_strategy", "bounded_random_search"),
                trial_budget=int(parsed.get("trial_budget", 5)),
                primary_metric=primary_metric,
                secondary_metrics=parsed.get("secondary_metrics", [m for m in allowed_metrics if m != primary_metric]),
                xai_methods=parsed.get("xai_methods", ["shap_tree_explainer", "permutation_importance"]),
                sensitivity_mode=parsed.get("sensitivity_mode", "both"),
                hyperparameters=parsed.get("hyperparameters", {}),
                rationale=parsed.get("architectural_rationale", "Plan proposed by OpenAI gpt-4.1."),
                trace_record=trace,
            )
            return plan

        except Exception as exc:
            logger.error("GPT-4.1 planning call failed: %s", exc)
            trace.schema_valid = False
            trace.latency_seconds = time.perf_counter() - t0
            trace.response_raw = str(exc)

            det_plan = DeterministicPolicyRunner().plan(domain, dataset_meta)
            det_plan.policy = "gpt41"
            det_plan.rationale = f"GPT-4.1 call failed ({exc}); defaulted through deterministic engine."
            det_plan.trace_record = trace
            return det_plan

    def _validate_plan(self, data: dict[str, Any], allowed_models: list[str], allowed_metrics: list[str]) -> bool:
        if not isinstance(data, dict):
            return False
        if "selected_model" not in data or data["selected_model"] not in allowed_models:
            return False
        if "primary_metric" not in data or data["primary_metric"] not in allowed_metrics:
            return False
        if "preprocessing" not in data or not isinstance(data["preprocessing"], dict):
            return False
        return True
