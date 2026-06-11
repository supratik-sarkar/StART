"""Interactive propensity-style model review demo.

Run interactively:        python examples/propensity_interactive.py
Run with safe defaults:   python examples/propensity_interactive.py --non-interactive

Flow: public sklearn binary-classification data (framed as a client
attrition / propensity case) -> stratified 60/20/20 train/test/OOS split ->
feature engineering checks -> choose model (Random Forest / XGBoost /
LightGBM) -> choose tuning (none / grid / random / Optuna) with five standard
hyperparameters per model -> choose holdout or K-fold (K=3 default, K=5) ->
fit -> train/test/OOS metrics table (AUC-ROC, Accuracy, Precision, Recall,
F1, top 10% lift) -> SHAP (or honest permutation fallback) explainability ->
parallel -30%..+30% shocks to top-5 features with AUC drift -> proof-carrying
report + tamper-evident ledger.

Equivalent CLI: `start propensity-demo` / `start propensity-demo --non-interactive`.
"""

from __future__ import annotations

import argparse

from start.modeling.propensity import PropensityOptions, prompt_options, run_propensity_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-interactive", action="store_true", help="Safe defaults, no prompts.")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--tuning", default="none")
    parser.add_argument("--cv", type=int, default=None, help="K for K-fold CV (3 or 5).")
    parser.add_argument("--cohort", default="test", help="Sensitivity cohort: test|oos|development.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    opts = PropensityOptions(
        model=args.model,
        tuning=args.tuning,
        cv_folds=args.cv,
        sensitivity_cohort=args.cohort,
        seed=args.seed,
    )
    if not args.non_interactive:
        opts = prompt_options(initial=opts)
    run_propensity_demo(opts)


if __name__ == "__main__":
    main()
