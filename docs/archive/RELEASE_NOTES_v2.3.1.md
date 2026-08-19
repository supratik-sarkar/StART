# StART v2.3.1 — Review Committee Polish

v2.3.1 is a cosmetic + workflow-maturity release on top of v2.3.0. It sharpens
the public wording, makes the review surfaces clearer and more transparent, and
adds real stratified K-fold tuning for the tabular suite. Core modeling and the
evidence/committee/MRM architecture are unchanged except where required for the
new K-fold artifacts.

## Highlights

### Public wording: "AI review committee"
The interactive review now reads as an AI review committee, not "enterprise
review":
- "Use AI review committee workflow? [Y/n]"
- "Running AI REVIEW COMMITTEE workflow"
- "AI review committee complete"
"Enterprise" now refers only to the private/firm LLM gateway path; the
``enterprise_llm_gateway`` provider name is unchanged.

### Dataset & target transparency
After loading, the review prints the dataset name, source, public URL, and
row/column count (for the demo: scikit-learn breast-cancer, UCI diagnostic
dataset). The target is shown with whether it was user-supplied or inferred by
discovery, plus the candidate target columns.

### Boxed panels + colored agent names
The committee roster and the LLM activation view render as two separate Rich
panels. Agent names use stable, distinct terminal colors (no plain white).

### AI Engineering Environment table
The adapter view is a colored Rich table — Adapter, Status, Purpose, Runtime,
Artifacts, Evidence, and an Install/fallback note when an adapter is
unavailable. Adapter names use distinct color styling. Mirrored in notebook 05.

### Safe endpoint display
The activation view shows an Endpoint for OpenAI, Anthropic, Grok, and the
enterprise gateway. For the gateway it shows the configured private-package
name and, only if the firm explicitly exposes one (via
``START_ENTERPRISE_LLM_ENDPOINT_PUBLIC``), a public endpoint; otherwise it shows
"private-package route (<package>); endpoint hidden". Secrets are never shown.

### Honest progress display
Heavy, counted work (e.g. the adapter sweep) shows a horizontal Rich progress
bar with a real percentage; work without an observable count uses an
indeterminate spinner with elapsed time. Progress is never faked, and the
helpers degrade to no-ops in non-interactive/batch runs.

### Review decision ledger
A compact decision ledger appears in the terminal, dashboard, transcript, and
notebook: checkpoint, recommendation, user choice, accepted/overridden/rejected
status, evidence, and a plain-language execution impact (e.g. rejecting
correlation pruning -> "kept all features"). Covers architecture, scaling,
outlier handling, correlation pruning, metric priority, validation, and
sign-off.

### Calibration maturity
The MRM sign-off's calibration (ECE) threshold is now configurable
(``GovernanceConfig.max_ece``, default 0.10) and the wording explains it is an
adjustable threshold per model and risk appetite, not a universal constant. The
calibration factor and its threshold appear in the dashboard/transcript.

### Real stratified K-fold tuning (tabular)
Tabular classification now uses real stratified K-fold model selection:
- Folds (default 5, configurable) are created ONLY inside the training split.
  The orchestrator reproduces the exact train/test/OOS split and passes only the
  training rows to K-fold, so test/OOS rows are never used for model selection.
- The primary metric is routed from the reviewer's cost priority (false
  negatives -> PR-AUC, false positives -> precision, balanced -> ROC-AUC).
- Artifacts: ``fold_metrics.csv``, ``tuning_trials.csv``, ``tuning_summary.json``.
- A terminal Rich table shows per-fold metrics, mean, std, and the selected
  params; the summary and fold metrics are included in the dashboard, transcript,
  and notebook.

The DL tabular path keeps single-split validation and is explicitly labelled
"Hyperparameter tuning (single-split validation)" — it never claims K-fold.

## Compatibility
All v2.3.0 behavior is preserved; v2.3.1 is additive. Existing tuning behavior
is unchanged when K-fold is not applicable, and the public clean-room
constraints (no firm name, endpoint, credential, or local path) still hold.
