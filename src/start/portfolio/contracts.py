"""Institutional portfolio analytical contracts, schemas, and threshold provenance.

Core architectural principle:
- PortfolioSpec remains the canonical owner of portfolio state.
- Analytical engines emit typed, immutable results with clear determinism classifications.
- Every threshold carries explicit provenance; no arbitrary numbers are invented as institutional truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class ThresholdSource(StrEnum):
    """Provenance origin for any numerical threshold or boundary test."""

    USER_POLICY = "user_policy"
    REGULATORY_RULE = "regulatory_rule"
    TEST_SPECIFICATION = "test_specification"
    MODEL_CONTRACT = "model_contract"
    SCENARIO_PARAMETER = "scenario_parameter"
    PRE_REGISTERED_VALIDATION = "pre_registered_validation"
    NONE = "none"


@dataclass(frozen=True)
class ThresholdValue:
    """A numerical threshold bound with explicit provenance."""

    metric: str
    warn: float | None = None
    fail: float | None = None
    source: ThresholdSource = ThresholdSource.NONE
    description: str = ""


class DeterminismTier(StrEnum):
    """Mathematical determinism classification."""

    EXACT_DETERMINISTIC = "exact_deterministic"
    NUMERICALLY_DETERMINISTIC = "numerically_deterministic_with_tolerance"
    SEEDED_STOCHASTIC = "seeded_stochastic"
    SEMANTIC_ARTIFACT = "semantic_artifact_determinism"
    LLM_EVIDENCE_CONSTRAINED = "llm_evidence_constrained"


@dataclass(frozen=True)
class MethodApplicability:
    """Applicability requirements and mathematical assumptions for a portfolio engine."""

    method_name: str
    required_inputs: tuple[str, ...]
    min_assets: int = 2
    min_observations: int = 2
    requires_psd_covariance: bool = True
    supports_missing_data: bool = False
    supports_bounds: bool = True
    supports_group_constraints: bool = False
    supports_turnover_constraints: bool = False
    determinism: DeterminismTier = DeterminismTier.NUMERICALLY_DETERMINISTIC
    assumptions: tuple[str, ...] = ()
    unsupported_combinations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioWeights:
    """Immutable asset weights vector with concentration diagnostics."""

    weights: dict[str, float]
    assets: tuple[str, ...]
    weights_sum: float
    max_weight: float
    min_weight: float
    herfindahl: float
    effective_n_positions: float
    gross_leverage: float
    n_active_positions: int

    @classmethod
    def from_series(cls, series: pd.Series) -> PortfolioWeights:
        assets = tuple(str(c) for c in series.index)
        values = series.to_numpy(dtype=float)
        weights_dict = {str(k): float(v) for k, v in series.items()}
        h = float(np.sum(values**2))
        eff_n = float(1.0 / h) if h > 1e-12 else 0.0
        return cls(
            weights=weights_dict,
            assets=assets,
            weights_sum=round(float(values.sum()), 10),
            max_weight=round(float(values.max()), 10) if len(values) else 0.0,
            min_weight=round(float(values.min()), 10) if len(values) else 0.0,
            herfindahl=round(h, 10),
            effective_n_positions=round(eff_n, 6),
            gross_leverage=round(float(np.sum(np.abs(values))), 10),
            n_active_positions=int((np.abs(values) > 1e-8).sum()),
        )


@dataclass(frozen=True)
class RiskContributionResult:
    """Euler variance and volatility risk contributions at asset and cluster level."""

    portfolio_variance: float
    portfolio_volatility: float
    marginal_contributions: dict[str, float]  # MCR_i = (Sigma w)_i / sigma_p
    component_contributions: dict[str, float]  # CR_i = w_i * MCR_i
    percentage_contributions: dict[str, float]  # %CR_i = CR_i / sigma_p = w_i (Sigma w)_i / sigma_p^2
    euler_reconciliation_error: float
    cluster_contributions: dict[str, float] = field(default_factory=dict)
    cluster_percentage_contributions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchicalTreeResult:
    """Typed representation of the hierarchical clustering tree and quasi-diagonal seriation."""

    assets: tuple[str, ...]
    distance_method: str
    linkage_method: str
    linkage_matrix: list[list[float]]  # (N-1, 4) linkage matrix
    leaf_order: tuple[int, ...]
    quasi_diagonal_order: tuple[str, ...]
    cluster_tree: dict[str, Any]
    correlation_fingerprint: str
    covariance_fingerprint: str
    cophenetic_correlation: float | None = None


@dataclass(frozen=True)
class LinkageSensitivityResult:
    """Sensitivity evaluation across hierarchical clustering linkages."""

    methods_compared: tuple[str, ...]
    weights_by_linkage: dict[str, dict[str, float]]
    effective_positions_by_linkage: dict[str, float]
    portfolio_variance_by_linkage: dict[str, float]
    pairwise_l1_distances: dict[str, float]
    pairwise_l2_distances: dict[str, float]
    max_asset_weight_diffs: dict[str, float]
    spearman_order_correlations: dict[str, float]


@dataclass(frozen=True)
class BootstrapStabilityResult:
    """Seeded time-series bootstrap cluster stability diagnostic."""

    bootstrap_method: str  # e.g. "stationary_block_bootstrap"
    block_size: int
    n_replicates: int
    seed: int
    assets: tuple[str, ...]
    pairwise_co_clustering_matrix: list[list[float]]
    mean_pairwise_stability: float
    min_pairwise_stability: float
    cophenetic_stability_mean: float


@dataclass(frozen=True)
class CopheneticResult:
    """Cophenetic correlation distance preservation diagnostic."""

    cophenetic_correlation: float
    linkage_method: str
    distance_method: str
    n_assets: int


@dataclass(frozen=True)
class EqualRiskContributionResult:
    """Equal Risk Contribution / Risk Parity solution."""

    weights: dict[str, float]
    risk_contributions: dict[str, float]
    percentage_risk_contributions: dict[str, float]
    target_risk_contribution: float  # 1 / N
    max_risk_contribution_dispersion: float  # max |%CR_i - 1/N|
    portfolio_volatility: float
    portfolio_variance: float
    objective_value: float
    solver_iterations: int
    solver_status: int
    converged: bool
    constraint_violations: dict[str, float]


@dataclass(frozen=True)
class FrontierPoint:
    """A single point along the efficient frontier."""

    label: str
    target_return: float
    expected_return_annualised: float
    volatility_annualised: float
    sharpe_annualised: float | None
    weights: dict[str, float]
    is_feasible: bool = True


@dataclass(frozen=True)
class EfficientFrontierResult:
    """Parametric efficient frontier curve and reference portfolio overlays."""

    frontier_points: tuple[FrontierPoint, ...]
    min_variance_point: FrontierPoint
    max_sharpe_point: FrontierPoint
    equal_weight_point: FrontierPoint | None = None
    erc_point: FrontierPoint | None = None
    hrp_point: FrontierPoint | None = None
    current_point: FrontierPoint | None = None


@dataclass(frozen=True)
class MethodComparisonResult:
    """Deterministic comparison matrix across portfolio construction methods."""

    methods: tuple[str, ...]
    summary_table: list[dict[str, Any]]
    weights_matrix: dict[str, dict[str, float]]
    risk_contributions_matrix: dict[str, dict[str, float]]


@dataclass(frozen=True)
class WalkForwardResult:
    """Non-leaky walk-forward portfolio evaluation result."""

    method: str
    rebalance_dates: tuple[str, ...]
    out_of_sample_returns: list[float]
    cumulative_returns: list[float]
    annualised_return: float
    annualised_volatility: float
    realized_sharpe: float | None
    max_drawdown: float
    mean_one_way_turnover: float
    transaction_cost_bps: float


# =========================================================================== #
# GATE-3 / GATE-3A INSTITUTIONAL CONSTRAINTS & ALLOCATION CONTRACTS
# =========================================================================== #
class MetricHorizon(StrEnum):
    """Temporal compounding horizon of a financial quantity."""

    PERIODIC = "PERIODIC"
    ANNUAL = "ANNUAL"
    SCENARIO_HORIZON = "SCENARIO_HORIZON"


@dataclass(frozen=True)
class ReturnConvention:
    """Explicit financial unit, frequency, and compounding convention."""

    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    frequency: str | None = None
    periods_per_year: float = 252.0
    is_annualised: bool = False
    return_basis: str = "simple"  # simple | log
    provenance: str = "explicit_contract"

    def validate_alignment(
        self,
        other: ReturnConvention,
        label_self: str = "first input",
        label_other: str = "second input",
    ) -> None:
        """Reject mismatches between periodic and annualized financial inputs."""
        h1 = MetricHorizon(self.horizon) if isinstance(self.horizon, str) else self.horizon
        h2 = MetricHorizon(other.horizon) if isinstance(other.horizon, str) else other.horizon
        if h1 != h2:
            raise ValueError(
                f"Financial horizon mismatch: {label_self} is {h1.value} but {label_other} is {h2.value}. "
                f"Mixing periodic and annualized inputs is rejected (fail-closed)."
            )
        if self.is_annualised != other.is_annualised:
            raise ValueError(
                f"Annualization mismatch: {label_self} is_annualised={self.is_annualised} but "
                f"{label_other} is_annualised={other.is_annualised}. (fail-closed)"
            )


def validate_horizon_alignment(
    mu_horizon: MetricHorizon | str | None,
    cov_horizon: MetricHorizon | str | None,
    periods_per_year: float = 252.0,
    frequency: str | None = None,
) -> None:
    """Validate horizon consistency across expected returns and covariance matrix."""
    if mu_horizon is not None and cov_horizon is not None:
        h_mu = MetricHorizon(mu_horizon) if isinstance(mu_horizon, str) else mu_horizon
        h_cov = MetricHorizon(cov_horizon) if isinstance(cov_horizon, str) else cov_horizon
        if h_mu != h_cov:
            raise ValueError(
                f"Financial horizon mismatch: expected returns mu has horizon {h_mu.value} while "
                f"covariance has horizon {h_cov.value}. Combining mismatched horizons is rejected (fail-closed)."
            )
        if h_mu == MetricHorizon.ANNUAL and h_cov == MetricHorizon.ANNUAL and periods_per_year != 1.0:
            raise ValueError(
                f"Double annualization error: inputs are already ANNUAL but periods_per_year={periods_per_year} "
                f"(expected periods_per_year=1.0 for already-annual inputs) (fail-closed)."
            )

    # Invariant: 252 periods/year alone must NOT semantically imply DAILY unless frequency metadata explicitly says daily
    if periods_per_year == 252.0 and frequency is not None and frequency not in ("daily", "business_daily"):
        raise ValueError(
            f"Frequency contradiction: periods_per_year=252.0 specified but frequency={frequency!r} is not daily."
        )


class ChallengeState(StrEnum):
    """Lifecycle state of an adversarial challenge."""

    OPEN = "OPEN"
    TOOL_REQUESTED = "TOOL_REQUESTED"
    EVIDENCE_GENERATED = "EVIDENCE_GENERATED"
    RESOLVED_NO_BREACH = "RESOLVED_NO_BREACH"
    RESOLVED_FINDING = "RESOLVED_FINDING"
    RESOLVED_EVIDENCE_ONLY = "RESOLVED_EVIDENCE_ONLY"
    BLOCKED = "BLOCKED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ChallengeResolution:
    """Audit-grade resolution record for an adversarial challenge."""

    challenge_id: str
    status: ChallengeState | str
    tool_name: str | None = None
    source_evidence_ids: tuple[str, ...] = ()
    generated_evidence_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    waiver_provenance: str | None = None
    decision_criterion: str = "NONE"  # USER_POLICY | MODEL_CONTRACT | REGULATORY_RULE | NONE
    decision_threshold: Any = None
    tool_request: str | None = None
    tool_parameters: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class ViewUncertaintyPolicy(StrEnum):
    """Specification policy for Black-Litterman view uncertainty matrix Omega."""

    EXPLICIT_OMEGA = "EXPLICIT_OMEGA"
    PROPORTIONAL_TAU_SIGMA = "PROPORTIONAL_TAU_SIGMA"
    IDZOREK_USER_CONFIDENCE = "IDZOREK_USER_CONFIDENCE"


class UncertaintyDerivationPolicy(StrEnum):
    """Specification policy for Robust MVO expected-return uncertainty covariance Sigma_mu."""

    EXPLICIT_UNCERTAINTY_COV = "EXPLICIT_UNCERTAINTY_COV"
    SAMPLE_COVARIANCE_DIV_N = "SAMPLE_COVARIANCE_DIV_N"
    IDENTITY_ESTIMATION = "IDENTITY_ESTIMATION"
    RESAMPLED_BOOTSTRAP = "RESAMPLED_BOOTSTRAP"


class ConstraintType(StrEnum):
    """Categorical type for portfolio optimization constraints."""

    BUDGET = "budget"
    LONG_ONLY = "long_only"
    MIN_WEIGHT = "min_weight"
    MAX_WEIGHT = "max_weight"
    GROSS_LEVERAGE = "gross_leverage"
    MAX_TURNOVER = "max_turnover"
    MAX_CONCENTRATION = "max_concentration"
    TRACKING_ERROR = "tracking_error"
    FACTOR_EXPOSURE = "factor_exposure"
    GROUP_EXPOSURE = "group_exposure"
    TRANSACTION_COST = "transaction_cost"


@dataclass(frozen=True)
class ConstraintViolation:
    """Individual constraint check record with explicit observed vs required bounds."""

    constraint: str
    observed_value: float
    required_bound: float | tuple[float, float]
    violation: float
    tolerance: float
    provenance: str
    status: str  # "SATISFIED" | "VIOLATED" | "WARNING"


@dataclass(frozen=True)
class ConstraintVerificationResult:
    """Comprehensive post-solve constraint verification report."""

    is_valid: bool
    max_violation: float
    tolerance: float
    violations: tuple[ConstraintViolation, ...]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionCostSpec:
    """Explicit transaction cost parameters per asset or asset class."""

    cost_bps: dict[str, float] = field(default_factory=dict)
    bid_ask_spread_bps: dict[str, float] = field(default_factory=dict)
    fixed_cost_bps: float = 0.0
    default_linear_bps: float = 0.0
    provenance: str = "COST_NOT_SUPPLIED"

    def get_asset_linear_bps(self, asset: str) -> float:
        if asset in self.cost_bps:
            return float(self.cost_bps[asset])
        return float(self.default_linear_bps)


@dataclass(frozen=True)
class FactorConstraintSpec:
    """Factor exposure constraint bounds."""

    factor_names: tuple[str, ...]
    loadings: dict[str, dict[str, float]]  # {asset: {factor: loading}}
    lower_bounds: dict[str, float] = field(default_factory=dict)
    upper_bounds: dict[str, float] = field(default_factory=dict)
    benchmark_relative: bool = False
    provenance: str = "user_factor_spec"

    def validate_asset_coverage(self, assets: list[str] | tuple[str, ...]) -> None:
        """Verify that every asset in the universe has explicit factor loadings."""
        missing = [a for a in assets if a not in self.loadings]
        if missing:
            raise ValueError(f"Asset(s) {missing} missing factor exposure in factor_constraints loadings (fail-closed)")
        for a in assets:
            for f in self.factor_names:
                if f not in self.loadings[a]:
                    raise ValueError(f"Asset '{a}' missing factor exposure for factor '{f}' (fail-closed)")


class GroupCoveragePolicy(StrEnum):
    """Institutional policy for group/sector constraint coverage."""

    OPTIONAL_UNMAPPED_ALLOWED = "OPTIONAL_UNMAPPED_ALLOWED"
    EXHAUSTIVE_PARTITION = "EXHAUSTIVE_PARTITION"
    DISJOINT_GROUPS_REQUIRED = "DISJOINT_GROUPS_REQUIRED"


@dataclass(frozen=True)
class GroupConstraintSpec:
    """Group/sector classification constraint bounds."""

    group_name: str
    memberships: dict[str, tuple[str, ...]]  # {group_label: (asset1, asset2, ...)}
    lower_bounds: dict[str, float] = field(default_factory=dict)  # {group_label: min_sum}
    upper_bounds: dict[str, float] = field(default_factory=dict)  # {group_label: max_sum}
    coverage_policy: GroupCoveragePolicy | str = GroupCoveragePolicy.OPTIONAL_UNMAPPED_ALLOWED
    allow_overlapping: bool = True
    provenance: str = "user_group_spec"

    def validate_asset_coverage(
        self,
        assets: list[str] | tuple[str, ...],
        coverage_policy: GroupCoveragePolicy | str | None = None,
        allow_unmapped: bool | None = None,
    ) -> None:
        """Validate group memberships against universe under explicit coverage policy."""
        all_members = {m for members in self.memberships.values() for m in members}
        unknown = all_members - set(assets)
        if unknown:
            raise ValueError(f"Group memberships contain unknown assets: {unknown} (fail-closed)")

        eff_policy = coverage_policy or self.coverage_policy
        if isinstance(eff_policy, str):
            eff_policy = GroupCoveragePolicy(eff_policy)

        # Check exhaustive partition / unmapped assets
        is_exhaustive = (
            eff_policy == GroupCoveragePolicy.EXHAUSTIVE_PARTITION
            or (allow_unmapped is False)
        )
        if is_exhaustive:
            unmapped = set(assets) - all_members
            if unmapped:
                raise ValueError(
                    f"Unmapped asset(s) {unmapped} not permitted under {eff_policy.value} coverage policy (fail-closed)"
                )

        # Check disjoint groups / overlapping
        is_disjoint = (
            eff_policy == GroupCoveragePolicy.DISJOINT_GROUPS_REQUIRED
            or (self.allow_overlapping is False)
        )
        if is_disjoint:
            seen: set[str] = set()
            overlapping: set[str] = set()
            for members in self.memberships.values():
                for m in members:
                    if m in seen:
                        overlapping.add(m)
                    seen.add(m)
            if overlapping:
                raise ValueError(
                    f"Overlapping group membership for asset(s) {overlapping} not permitted under "
                    f"{eff_policy.value} policy (fail-closed)"
                )


@dataclass(frozen=True)
class RebalanceDecision:
    """Analytical rebalance decision object containing trade weights and net risk/cost metrics."""

    current_weights: dict[str, float]
    proposed_weights: dict[str, float]
    trade_weights: dict[str, float]
    turnover: float
    estimated_transaction_cost: float
    constraint_verification: ConstraintVerificationResult
    pre_trade_risk: dict[str, float]
    post_trade_risk: dict[str, float]
    expected_return_gross_periodic: float | None = None
    expected_return_gross_annualised: float | None = None
    expected_return_net_periodic: float | None = None
    expected_return_net_annualised: float | None = None
    cost_provenance: str = "COST_NOT_SUPPLIED"
    evidence_ids: tuple[str, ...] = ()
    expected_return_gross: float | None = None  # legacy compat
    expected_return_net: float | None = None  # legacy compat


@dataclass(frozen=True)
class BlackLittermanResult:
    """Mathematically explicit Black-Litterman model result."""

    implied_returns: dict[str, float]
    posterior_returns: dict[str, float]
    prior_weights: dict[str, float]
    posterior_weights: dict[str, float]
    view_residuals: dict[str, float]
    view_uncertainties: dict[str, float]
    risk_aversion: float
    tau: float
    turnover_vs_prior: float
    constraint_verification: ConstraintVerificationResult
    posterior_covariance_fingerprint: str
    p_matrix: list[list[float]]
    q_vector: list[float]
    omega_matrix: list[list[float]]
    view_labels: tuple[str, ...]
    posterior_volatility_annualised: float
    posterior_sharpe_annualised: float | None = None
    uncertainty_policy: ViewUncertaintyPolicy | str = ViewUncertaintyPolicy.PROPORTIONAL_TAU_SIGMA
    converged: bool = True
    usable_solution: bool = True
    solver_status: str = "OPTIMAL"
    solver_message: str = ""


@dataclass(frozen=True)
class RobustMVOResult:
    """Convex Robust Mean-Variance Optimization result under ellipsoidal uncertainty."""

    weights: dict[str, float]
    uncertainty_radius: float
    nominal_expected_return_annualised: float
    worst_case_expected_return_annualised: float
    portfolio_volatility_annualised: float
    nominal_sharpe_annualised: float | None
    worst_case_sharpe_annualised: float | None
    effective_n_positions: float
    turnover_vs_prior: float | None
    constraint_verification: ConstraintVerificationResult
    uncertainty_set_type: str = "ellipsoidal_return"
    uncertainty_policy: UncertaintyDerivationPolicy | str = UncertaintyDerivationPolicy.EXPLICIT_UNCERTAINTY_COV
    uncertainty_covariance_fingerprint: str = ""
    converged: bool = True
    usable_solution: bool = True
    solver_status: str = "OPTIMAL"
    solver_message: str = ""


@dataclass(frozen=True)
class RobustSensitivityPoint:
    """A single evaluation point along the robust optimization uncertainty radius grid."""

    uncertainty_radius: float
    nominal_expected_return_annualised: float
    worst_case_expected_return_annualised: float
    portfolio_volatility_annualised: float
    worst_case_sharpe_annualised: float | None
    effective_n_positions: float
    turnover_vs_prior: float | None
    weights: dict[str, float]


@dataclass(frozen=True)
class RobustSensitivityResult:
    """Grid sensitivity evaluation over uncertainty parameters."""

    points: tuple[RobustSensitivityPoint, ...]
    radii_evaluated: tuple[float, ...]
    baseline_radius: float


@dataclass(frozen=True)
class CVaROptimizationResult:
    """Rockafellar-Uryasev CVaR / Expected-Shortfall Linear Programming optimization result."""

    weights: dict[str, float]
    confidence_level: float
    cvar_at_scenario_horizon: float
    var_at_scenario_horizon: float
    tail_scenario_count: int
    n_scenarios: int
    expected_return_periodic: float
    effective_n_positions: float
    constraint_verification: ConstraintVerificationResult
    converged: bool
    usable_solution: bool
    solver_status: str
    scenario_horizon: str = "1_PERIOD"
    cvar_periodic: float = 0.0  # legacy compat
    cvar_annualised: float | None = None  # deprecated/explicit only
    var_auxiliary_periodic: float = 0.0  # legacy compat
    var_auxiliary_annualised: float | None = None  # deprecated/explicit only
    expected_return_annualised: float = 0.0
    turnover_vs_prior: float | None = None
    solver_message: str = ""


@dataclass(frozen=True)
class HERCResult:
    """Hierarchical Equal Risk Contribution (HERC) allocation result."""

    weights: dict[str, float]
    tree_result: HierarchicalTreeResult
    cluster_risk_contributions: dict[str, float]
    percentage_risk_contributions: dict[str, float]
    effective_n_positions: float
    portfolio_volatility_annualised: float
    portfolio_variance: float
    constraint_verification: ConstraintVerificationResult
    risk_measure: str = "variance"
    converged: bool = True
    usable_solution: bool = True
    solver_status: str = "OPTIMAL"


@dataclass(frozen=True)
class MaxDiversificationResult:
    """Maximum Diversification Portfolio result."""

    weights: dict[str, float]
    diversification_ratio: float
    weighted_asset_volatility_annualised: float
    portfolio_volatility_annualised: float
    effective_n_positions: float
    constraint_verification: ConstraintVerificationResult
    converged: bool = True
    usable_solution: bool = True
    solver_status: str = "OPTIMAL"
    solver_message: str = ""


@dataclass(frozen=True)
class TrackingErrorResult:
    """Benchmark-relative tracking-error constrained portfolio result."""

    weights: dict[str, float]
    benchmark_weights: dict[str, float]
    active_weights: dict[str, float]
    tracking_error_periodic: float
    tracking_error_annualised: float
    active_return_annualised: float | None
    information_ratio: float | None
    portfolio_volatility_annualised: float
    constraint_verification: ConstraintVerificationResult
    converged: bool = True
    usable_solution: bool = True
    solver_status: str = "OPTIMAL"
    solver_message: str = ""


# =========================================================================== #
# GATE 4: COVARIANCE, FACTOR RISK & ATTRIBUTION CONTRACTS
# =========================================================================== #
class PSDRepairMethod(StrEnum):
    """Institutional Positive Semi-Definite (PSD) matrix repair methods."""

    SPECTRAL_CLIPPING = "SPECTRAL_CLIPPING"
    HIGHAM_NEAREST_CORRELATION = "HIGHAM_NEAREST_CORRELATION"


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Deterministic structural and spectral diagnostics for covariance matrices."""

    n_assets: int
    is_symmetric: bool
    symmetry_error: float
    is_psd: bool
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    eigenvalue_spectrum: tuple[float, ...]
    rank: int
    numerical_rank: int
    condition_number: float
    trace: float
    log_determinant: float | None
    effective_rank: float
    largest_eigenvalue_share: float
    diagonal_positive: bool
    valid_correlation_conversion: bool
    matrix_fingerprint: str


@dataclass(frozen=True)
class PSDRepairResult:
    """Exact provenance record of a numerical PSD matrix repair intervention."""

    repair_method: PSDRepairMethod | str
    original_minimum_eigenvalue: float
    repaired_minimum_eigenvalue: float
    frobenius_distortion: float
    relative_frobenius_distortion: float
    maximum_element_change: float
    diagonal_preserved: bool
    iterations_used: int
    converged: bool
    matrix_fingerprint_before: str
    matrix_fingerprint_after: str
    repaired_matrix: list[list[float]]
    pd_floor: float = 0.0
    intervention_reason: str = "Non-PSD input requiring explicit numerical repair"


@dataclass(frozen=True)
class CovarianceEstimate:
    """Common institutional covariance estimate contract."""

    estimator: str
    asset_order: tuple[str, ...]
    matrix: list[list[float]]
    input_horizon: MetricHorizon | str
    frequency: str | None
    periods_per_year: float
    n_observations: int | None
    missing_data_policy: str
    parameters: dict[str, Any]
    diagnostics: CovarianceDiagnostics
    matrix_fingerprint: str
    estimation_provenance: str


@dataclass(frozen=True)
class CovarianceComparisonResult:
    """Deterministic comparative matrix evaluation across covariance estimators."""

    estimators_compared: tuple[str, ...]
    asset_order: tuple[str, ...]
    diagnostics_by_estimator: dict[str, CovarianceDiagnostics]
    pairwise_frobenius_distances: dict[str, float]
    pairwise_spectral_distances: dict[str, float]
    portfolio_volatilities_annualised: dict[str, float]
    portfolio_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class FactorRiskModelResult:
    """Linear Factor Risk Model representation: r = B f + epsilon, Sigma = B F B' + D."""

    asset_order: tuple[str, ...]
    factor_order: tuple[str, ...]
    exposure_matrix: list[list[float]]  # n_assets x n_factors
    factor_covariance: list[list[float]]  # n_factors x n_factors
    specific_variances: dict[str, float]  # asset -> specific variance
    reconstructed_covariance: list[list[float]]  # n_assets x n_assets
    exposure_fingerprint: str
    factor_covariance_fingerprint: str
    specific_variance_fingerprint: str
    reconstructed_covariance_fingerprint: str
    diagnostics: CovarianceDiagnostics
    time_alignment: str = "beginning_of_period_exposures"
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    frequency: str | None = None
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class FactorRiskDecompositionResult:
    """Euler-consistent systematic variance and factor risk component decomposition."""

    weights: dict[str, float]
    portfolio_factor_exposures: dict[str, float]  # factor -> b_p
    systematic_variance_periodic: float
    specific_variance_periodic: float
    total_variance_periodic: float
    portfolio_volatility_annualised: float
    systematic_volatility_annualised: float
    specific_volatility_annualised: float
    systematic_variance_share: float
    specific_variance_share: float
    factor_variance_contributions_periodic: dict[str, float]  # factor -> C_k = b_k * (Fb)_k
    factor_variance_shares: dict[str, float]  # factor -> C_k / V_total
    asset_specific_variance_contributions: dict[str, float]  # asset -> w_i^2 * d_i
    euler_reconciliation_error: float
    total_reconciliation_error: float
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class ActiveRiskDecompositionResult:
    """Benchmark-relative active risk (tracking error) factor and specific decomposition."""

    weights: dict[str, float]
    benchmark_weights: dict[str, float]
    active_weights: dict[str, float]  # a = w - w_b
    active_factor_exposures: dict[str, float]  # Delta b = B' (w - w_b)
    factor_active_variance_periodic: float  # Delta b' F Delta b
    specific_active_variance_periodic: float  # a' D a
    total_active_variance_periodic: float
    tracking_error_annualised: float
    factor_active_share: float
    specific_active_share: float
    active_factor_contributions_periodic: dict[str, float]
    asset_specific_active_contributions: dict[str, float]  # asset -> a_i^2 * d_i
    reconciliation_error: float
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class FactorReturnAttributionResult:
    """Exact period-by-period factor return performance attribution."""

    n_periods: int
    factor_order: tuple[str, ...]
    asset_order: tuple[str, ...]
    total_portfolio_return: float
    total_factor_contribution: float
    total_specific_contribution: float
    cumulative_factor_contributions: dict[str, float]
    period_portfolio_returns: tuple[float, ...]
    period_factor_contributions: tuple[dict[str, float], ...]
    period_specific_contributions: tuple[float, ...]
    period_reconciliation_errors: tuple[float, ...]
    max_abs_reconciliation_error: float
    mean_abs_reconciliation_error: float
    time_alignment_convention: str
    is_reconciled: bool


@dataclass(frozen=True)
class BrinsonAttributionResult:
    """Single-period Brinson-Fachler active performance attribution."""

    group_names: tuple[str, ...]
    portfolio_group_weights: dict[str, float]
    benchmark_group_weights: dict[str, float]
    portfolio_group_returns: dict[str, float]
    benchmark_group_returns: dict[str, float]
    total_portfolio_return: float
    total_benchmark_return: float
    total_active_return: float
    allocation_effects: dict[str, float]  # A_g = (w_pg - w_bg) * (r_bg - R_b)
    selection_effects: dict[str, float]  # S_g = w_bg * (r_pg - r_bg)
    interaction_effects: dict[str, float]  # I_g = (w_pg - w_bg) * (r_pg - r_bg)
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    reconciliation_error: float
    convention: str = "BRINSON_FACHLER"
    is_reconciled: bool = True


@dataclass(frozen=True)
class CarinoLinkedAttributionResult:
    """Multi-period geometric active attribution linking via Carino logarithmic smoothing."""

    n_periods: int
    group_names: tuple[str, ...]
    total_portfolio_return_geometric: float
    total_benchmark_return_geometric: float
    total_active_return_geometric: float
    linked_allocation_effects: dict[str, float]
    linked_selection_effects: dict[str, float]
    linked_interaction_effects: dict[str, float]
    total_linked_allocation: float
    total_linked_selection: float
    total_linked_interaction: float
    period_linking_coefficients: tuple[float, ...]  # k_t
    benchmark_linking_coefficient: float  # K
    reconciliation_error: float
    is_reconciled: bool = True


@dataclass(frozen=True)
class FactorDataIntegrityResult:
    """Deterministic pre-flight verification of factor model data alignment and integrity."""

    is_valid: bool
    n_assets: int
    n_factors: int
    assets: tuple[str, ...]
    factors: tuple[str, ...]
    has_duplicate_assets: bool
    has_duplicate_factors: bool
    missing_exposure_count: int
    missing_factor_return_count: int
    missing_specific_variance_count: int
    has_lookahead_violation: bool
    issues: tuple[str, ...]


# =========================================================================== #
# GATE 5: TAIL RISK, EXPECTED SHORTFALL & ADVANCED BACKTESTING CONTRACTS
# =========================================================================== #
class TailSignConvention(StrEnum):
    """Loss and return sign convention for tail risk metrics."""

    POSITIVE_LOSS_MAGNITUDE = "positive_loss_magnitude"
    NEGATIVE_RETURN_QUANTILE = "negative_return_quantile"


class TailRiskMethod(StrEnum):
    """Estimation methodology for VaR and Expected Shortfall."""

    HISTORICAL = "historical"
    PARAMETRIC_NORMAL = "parametric_normal"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True)
class TailRiskEstimate:
    """Quantitative VaR and Expected Shortfall estimate with exact finite-sample tail mass provenance."""

    method: TailRiskMethod | str
    confidence: float  # alpha_var, e.g. 0.95, 0.99
    sign_convention: TailSignConvention | str
    var: float  # Positive loss magnitude
    es: float  # Positive loss magnitude
    n_observations: int
    tail_observations_count: int
    tail_fraction: float
    boundary_weight: float  # Fractional weight applied to boundary observation in exact tail average
    quantile_method: str  # e.g., 'linear', 'lower', 'higher', 'nearest'
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    frequency: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    converged: bool = True
    limitations: tuple[str, ...] = ()
    data_fingerprint: str = ""


@dataclass(frozen=True)
class TailBacktestResult:
    """Comprehensive out-of-sample exception backtest combining Kupiec and Christoffersen diagnostics."""

    pnl_source: str  # actual | hypothetical
    var_confidence: float  # alpha_var (e.g. 0.99, null probability p0 = 1 - alpha_var)
    test_significance: float  # gamma_test (e.g. 0.05)
    n_observations: int
    n_exceptions: int
    exception_rate: float
    expected_probability: float
    expected_exceptions: float
    kupiec_lr: float
    kupiec_p_value: float
    kupiec_rejected: bool
    kupiec_estimable: bool
    christoffersen_lr: float
    christoffersen_p_value: float
    christoffersen_rejected: bool
    christoffersen_estimable: bool
    conditional_coverage_lr: float
    conditional_coverage_p_value: float
    conditional_coverage_rejected: bool
    conditional_coverage_estimable: bool
    transition_counts: tuple[int, int, int, int]  # (n00, n01, n10, n11)
    pi_01: float | None
    pi_11: float | None
    has_zero_transition_cell: bool
    exception_dates: tuple[str, ...] = ()
    indicators: tuple[int, ...] = ()
    indicator_hash: str = ""
    exception_convention: str = "I_t = 1 iff loss_t > VaR_t (PnL_t < -VaR_t)"
    limitations: tuple[str, ...] = ()
    data_fingerprint: str = ""

    @property
    def n00(self) -> int:
        return self.transition_counts[0]

    @property
    def n01(self) -> int:
        return self.transition_counts[1]

    @property
    def n10(self) -> int:
        return self.transition_counts[2]

    @property
    def n11(self) -> int:
        return self.transition_counts[3]


@dataclass(frozen=True)
class DurationDiagnosticsResult:
    """Descriptive statistics on inter-exception durations and clustering."""

    n_durations: int
    mean_duration: float
    median_duration: float
    min_duration: int
    max_duration: int
    duration_std: float
    max_run_length: int
    durations: tuple[int, ...] = ()
    data_fingerprint: str = ""


@dataclass(frozen=True)
class TailSeverityResult:
    """Exceedance loss and tail severity magnitude analysis for exception events."""

    n_exceptions: int
    mean_absolute_exceedance: float
    median_absolute_exceedance: float
    max_absolute_exceedance: float
    total_tail_exceedance_loss: float
    mean_normalized_exceedance: float | None
    max_normalized_exceedance: float | None
    mean_relative_exceedance: float | None
    max_relative_exceedance: float | None
    absolute_exceedances: tuple[float, ...] = ()
    data_fingerprint: str = ""


@dataclass(frozen=True)
class TailRiskContributionResult:
    """Euler component and marginal decomposition of VaR and Expected Shortfall."""

    method: str
    confidence: float
    portfolio_var: float
    portfolio_es: float
    component_var: dict[str, float]  # Component VaR per asset
    component_es: dict[str, float]  # Component ES per asset
    percentage_var_contributions: dict[str, float]
    percentage_es_contributions: dict[str, float]
    var_reconciliation_error: float
    es_reconciliation_error: float
    data_fingerprint: str = ""


@dataclass(frozen=True)
class TailModelComparisonResult:
    """Multi-method comparison across historical and parametric tail risk models."""

    models_compared: tuple[str, ...]
    confidence: float
    estimates: dict[str, TailRiskEstimate]
    var_values: dict[str, float]
    es_values: dict[str, float]
    es_to_var_ratios: dict[str, float]
    data_fingerprint: str = ""


# =========================================================================== #
# GATE 6: INSTITUTIONAL SCENARIO, STRESS & REVERSE-STRESS CONTRACTS
# =========================================================================== #

class ScenarioType(StrEnum):
    """Categorical classification of stress and scenario projections."""

    SYNTHETIC = "SYNTHETIC"
    USER_DEFINED = "USER_DEFINED"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    POLICY_DEFINED = "POLICY_DEFINED"
    REVERSE_STRESS = "REVERSE_STRESS"


class ShockSpace(StrEnum):
    """Risk-factor space to which a scenario shock applies."""

    ASSET_RETURN = "ASSET_RETURN"
    FACTOR_RETURN = "FACTOR_RETURN"
    PRICE = "PRICE"
    YIELD = "YIELD"
    RATE = "RATE"
    SPREAD = "SPREAD"
    VOLATILITY = "VOLATILITY"
    FX_RATE = "FX_RATE"
    CUSTOM_RISK_FACTOR = "CUSTOM_RISK_FACTOR"


class ShockUnit(StrEnum):
    """Explicit measurement unit of the raw input shock."""

    RETURN_DECIMAL = "RETURN_DECIMAL"  # e.g., -0.10 for -10% return
    RELATIVE_PERCENT = "RELATIVE_PERCENT"  # e.g., -10.0 for -10% price shift
    BASIS_POINTS = "BASIS_POINTS"  # e.g., +100.0 for +0.01 rate change
    ABSOLUTE = "ABSOLUTE"  # e.g., +0.02
    LOG_RETURN = "LOG_RETURN"  # e.g., ln(P1/P0)
    VOLATILITY_POINTS = "VOLATILITY_POINTS"  # e.g., +5.0 for +5 vol percentage points


class RepricingMethod(StrEnum):
    """Mathematical revaluation methodology consumed by the portfolio stress engine."""

    LINEAR_RETURN = "LINEAR_RETURN"
    FACTOR_LINEAR = "FACTOR_LINEAR"
    DELTA = "DELTA"
    DELTA_GAMMA = "DELTA_GAMMA"
    FULL_REVALUATION_ADAPTER = "FULL_REVALUATION_ADAPTER"
    CUSTOM_DETERMINISTIC_ADAPTER = "CUSTOM_DETERMINISTIC_ADAPTER"


class ReverseStressNorm(StrEnum):
    """Distance norm / geometry for minimum shock reverse stress optimization."""

    L2 = "L2"
    WEIGHTED_L2 = "WEIGHTED_L2"
    MAHALANOBIS = "MAHALANOBIS"


class PartitionContract(StrEnum):
    """Group decomposition additivity contract."""

    EXHAUSTIVE_PARTITION = "EXHAUSTIVE_PARTITION"
    OVERLAPPING_ANALYTICAL = "OVERLAPPING_ANALYTICAL"


@dataclass(frozen=True)
class ScenarioShock:
    """Explicit typed leg representing a single risk-factor shock with raw and normalized provenance."""

    risk_factor_id: str
    shock_space: ShockSpace | str
    shock_unit: ShockUnit | str
    raw_value: float
    normalized_value: float
    normalization_rule: str
    computational_unit: str = "RETURN_DECIMAL"
    base_value: float | None = None
    currency: str | None = None
    source_reference: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensitivitySpec:
    """First- and second-order sensitivities (Delta and Gamma) for an instrument or risk factor."""

    risk_factor_id: str
    delta: float
    gamma: float = 0.0
    sensitivity_unit: str = "CURRENCY_PER_UNIT"
    as_of_date: str = ""
    source: str = ""
    base_state_fingerprint: str = ""


@dataclass(frozen=True)
class ScenarioSpec:
    """Evidence-bearing deterministic specification of a scenario."""

    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType | str
    shocks: tuple[ScenarioShock, ...]
    repricing_method: RepricingMethod | str
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC
    frequency: str | None = None
    as_of_date: str = ""
    source_reference: str = ""
    source_fingerprint: str = ""
    specific_shock_policy: str = "NONE"  # NONE | EXPLICIT_ZERO | SUPPLIED
    currency: str = ""
    assumptions: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    """Deterministic result of evaluating a portfolio under a scenario."""

    scenario_id: str
    scenario_type: str
    repricing_method: str
    scenario_return: float  # Portfolio return under scenario
    scenario_loss: float  # Canonical positive loss magnitude (-scenario_return)
    portfolio_value: float | None
    scenario_pnl: float | None  # portfolio_value * scenario_return
    scenario_monetary_loss: float | None  # -scenario_pnl
    asset_contributions: dict[str, float]
    factor_contributions: dict[str, float]
    specific_contribution: float | None
    group_contributions: dict[str, float]
    partition_contract: str
    reconciliation_error: float
    converged: bool
    limitations: tuple[str, ...] = ()
    data_fingerprint: str = ""
    horizon: str = ""
    currency: str = ""
    portfolio_state_fingerprint: str = ""


@dataclass(frozen=True)
class ActiveScenarioResult:
    """Scenario evaluation comparing portfolio, benchmark, and active portfolio returns."""

    scenario_id: str
    portfolio_return: float
    benchmark_return: float
    active_return: float
    portfolio_loss: float
    benchmark_loss: float
    active_loss: float
    active_asset_contributions: dict[str, float]
    active_factor_contributions: dict[str, float]
    reconciliation_error: float
    data_fingerprint: str = ""


@dataclass(frozen=True)
class ScenarioSetResult:
    """Multi-scenario comparative ranking across a set of heterogeneous scenarios."""

    scenarios_evaluated: tuple[str, ...]
    ranking_metric: str  # e.g., "scenario_loss"
    scenario_returns: dict[str, float]
    scenario_losses: dict[str, float]
    scenario_pnls: dict[str, float]
    loss_rankings: tuple[str, ...]  # Sorted descending by canonical scenario_loss
    worst_scenario_id: str
    best_scenario_id: str
    worst_scenario_loss: float
    best_scenario_loss: float
    method_disclosures: dict[str, str]
    comparability_valid: bool
    limitations: tuple[str, ...] = ()
    data_fingerprint: str = ""


@dataclass(frozen=True)
class ScenarioSensitivityPoint:
    """Single response point in a scenario parameter sensitivity sweep."""

    shock_multiplier: float
    raw_shock_value: float
    normalized_shock_value: float
    portfolio_return: float
    portfolio_loss: float
    portfolio_pnl: float | None


@dataclass(frozen=True)
class ScenarioSensitivityResult:
    """Response curve across deterministic parameter sweeps for a selected risk factor."""

    risk_factor_id: str
    grid_points: tuple[ScenarioSensitivityPoint, ...]
    base_loss: float
    max_loss: float
    min_loss: float
    data_fingerprint: str = ""


@dataclass(frozen=True)
class ReverseStressSpec:
    """Specification of a reverse-stress problem seeking the minimum shock achieving a target loss."""

    target_loss: float  # Canonical positive loss magnitude (> 0)
    shock_space: ShockSpace | str = ShockSpace.FACTOR_RETURN
    repricing_method: RepricingMethod | str = RepricingMethod.LINEAR_RETURN
    distance_norm: ReverseStressNorm | str = ReverseStressNorm.L2
    allowed_factors: tuple[str, ...] = ()
    shock_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    reference_covariance: list[list[float]] | None = None
    covariance: Any | None = None
    scaling_factors: dict[str, float] = field(default_factory=dict)
    weight_matrix: Any | None = None
    risk_factor_units: dict[str, str] = field(default_factory=dict)
    is_heterogeneous_unscaled: bool = False
    covariance_fingerprint: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReverseStressResult:
    """Deterministic result of solving a reverse-stress problem."""

    target_loss: float
    achieved_loss: float
    achieved_return: float
    loss_gap: float
    shock_vector: dict[str, float]
    normalized_shocks: dict[str, float]
    distance: float
    distance_norm: str
    bounds_satisfied: bool
    solver_status: str
    converged: bool
    is_closed_form: bool
    limitations: tuple[str, ...] = ()
    data_fingerprint: str = ""


@dataclass(frozen=True)
class ScenarioDataIntegrityResult:
    """Data integrity and semantic validation outcome for scenario inputs."""

    scenario_id: str
    valid: bool
    n_shocks: int
    shock_spaces_present: tuple[str, ...] = ()
    shock_units_present: tuple[str, ...] = ()
    repricing_compatible: bool = True
    sensitivities_complete: bool = True
    coverage_complete: bool = True
    provenance_valid: bool = True
    issues: tuple[str, ...] = ()
    data_fingerprint: str = ""




