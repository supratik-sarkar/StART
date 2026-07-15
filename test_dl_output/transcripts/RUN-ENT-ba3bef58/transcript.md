# Review committee transcript — RUN-ENT-ba3bef58

## Decisions

### Review decision ledger

| Checkpoint | Recommended | User choice | Status | Evidence | Execution impact |
| --- | --- | --- | --- | --- | --- |
| validation | accept | accept | accepted | — | validation review accepted into signoff |

## User overrides

_No overrides — user accepted all recommendations._

## Agent conversations

_No questions asked of the agents._

## ValidationAgent review

- Most sensitive feature: worst_perimeter
- Max |drift|: 0.001506
- Signoff impact: low feature dependence; no signoff concern from sensitivity.
- The model shows no material dependence on 'worst_perimeter' (max metric drift 0.0015 under +/-30% shocks). If 'worst_perimeter' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows no material dependence on 'worst_area' (max metric drift 0.0006 under +/-30% shocks). If 'worst_area' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows no material dependence on 'mean_perimeter' (max metric drift 0.0006 under +/-30% shocks). If 'mean_perimeter' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.

## MRM signoff decision

**Verdict: READY WITH CONDITIONS**

READY WITH CONDITIONS: 0 blocker(s), 1 concern(s), 5 factor(s) clear across performance, generalization, calibration, feature dependence, and reviewer activity. Sign-off is conditional on addressing the listed concerns.

| Factor | Status | Detail | Evidence |
| --- | --- | --- | --- |
| Performance | ok | OOS auc_roc=0.9671 | cohort_metrics.oos |
| Generalization | ok | train-OOS gap -0.0015 | cohort_metrics |
| Calibration | concern | OOS ECE=0.2570 exceeds the configured threshold 0.100 (adjustable per model/risk appetite) | cohort_metrics.oos.ece |
| Feature dependence | ok | max drift 0.0015 (most sensitive: worst_perimeter) | sensitivity_analysis |
| Reviewer challenges | ok | no outstanding reviewer challenges | review_session |
| Reviewer overrides | ok | no overrides; reviewer accepted recommendations | review_session |

## Sensitivity

### Sensitivity analysis

- Metric: auc_roc
- Baseline (0% shock): 0.9880
- Most sensitive feature: worst_perimeter
- Max |drift|: 0.0015

| Feature | Shock % | Baseline | Shocked | Delta | Risk impact |
| --- | --- | --- | --- | --- | --- |
| worst_area | -30% | 0.9880 | 0.9874 | -0.0006 | negligible |
| worst_area | -20% | 0.9880 | 0.9876 | -0.0004 | negligible |
| worst_area | -10% | 0.9880 | 0.9878 | -0.0002 | negligible |
| worst_area | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| worst_area | +10% | 0.9880 | 0.9882 | +0.0002 | negligible |
| worst_area | +20% | 0.9880 | 0.9884 | +0.0004 | negligible |
| worst_area | +30% | 0.9880 | 0.9885 | +0.0004 | negligible |
| mean_area | -30% | 0.9880 | 0.9876 | -0.0004 | negligible |
| mean_area | -20% | 0.9880 | 0.9878 | -0.0002 | negligible |
| mean_area | -10% | 0.9880 | 0.9880 | -0.0000 | negligible |
| mean_area | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| mean_area | +10% | 0.9880 | 0.9881 | +0.0001 | negligible |
| mean_area | +20% | 0.9880 | 0.9882 | +0.0002 | negligible |
| mean_area | +30% | 0.9880 | 0.9883 | +0.0003 | negligible |
| area_error | -30% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | -20% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | -10% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| area_error | +10% | 0.9880 | 0.9880 | +0.0000 | negligible |
| area_error | +20% | 0.9880 | 0.9881 | +0.0001 | negligible |

Most sensitive feature: worst_perimeter (max |drift| 0.0015 in auc_roc). Large drift indicates the model relies heavily on that feature; review for stability.

### K-fold tuning

- Method: stratified_kfold (5-fold, stratified)
- Primary metric: auc_roc
- Train rows used: 341 (test/OOS rows excluded from selection: 228)
- Best params: {'C': 3.0, 'class_weight': None}
- Best mean auc_roc: 0.9925 (std 0.0069)

| Fold | Metric (best params) | n_train | n_val |
| --- | --- | --- | --- |
| 1 | 0.9937 | 272 | 69 |
| 2 | 0.9799 | 273 | 68 |
| 3 | 0.9916 | 273 | 68 |
| 4 | 0.9991 | 273 | 68 |
| 5 | 0.9981 | 273 | 68 |

**Mean:** 0.9925  |  **Std:** 0.0069
