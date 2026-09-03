"""Institutional Scenario, Stress & Reverse-Stress Intelligence Engines.

Core Invariants:
1. Deterministic Computation: Scenario P&L, factor P&L, delta-gamma approximations,
   loss rankings, reverse-stress solutions, and sensitivity grids are computed strictly
   by deterministic engines (zero LLM arithmetic in prose).
2. Per-Shock Typed Contracts: Every shock explicitly states its risk factor ID, shock space,
   shock unit, raw value, and normalized value.
3. Raw vs Normalized Provenance: Raw user/policy inputs (e.g. +100 bps, -10%) are preserved
   traceably alongside normalized decimal values (+0.0100, -0.1000).
4. Method x Shock Compatibility: Repricing methods strictly validate shock spaces and fail closed
   on unsupported combinations (e.g. linear return on basis points without sensitivity).
5. Canonical Scenario Loss & Sign:
   - scenario_return: portfolio return under scenario.
   - scenario_loss: -scenario_return (canonical positive loss magnitude).
   - scenario_pnl: portfolio_value * scenario_return.
   - scenario_monetary_loss: -scenario_pnl.
   - Worst scenario in ranking: maximum canonical scenario_loss.
6. Delta-Gamma Formulation: P&L_approx = delta' Delta_x + 0.5 Delta_x' Gamma Delta_x using full
   symmetric Hessian matrix Gamma with symmetry validation.
7. Reverse Stress P0: Linear loss formulation c' x >= L* solved analytically (unconstrained L2)
   or via convex quadratic optimization (bounded, weighted L2, Mahalanobis) with rigorous post-solve
   verification of target loss achievement and bounds.
8. Historical Provenance: Historical replay requires explicit source references, observation windows,
   and explicit proxy mapping for unmapped assets (no implicit zero).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from start.portfolio.contracts import (
    ActiveScenarioResult,
    MetricHorizon,
    PartitionContract,
    RepricingMethod,
    ReverseStressNorm,
    ReverseStressResult,
    ReverseStressSpec,
    ScenarioDataIntegrityResult,
    ScenarioResult,
    ScenarioSensitivityPoint,
    ScenarioSensitivityResult,
    ScenarioSetResult,
    ScenarioShock,
    ScenarioSpec,
    ScenarioType,
    SensitivitySpec,
    ShockSpace,
    ShockUnit,
)


def _series_fingerprint(arr: Any) -> str:
    """Deterministic 32-character SHA-256 fingerprint for arrays or dicts."""
    if isinstance(arr, dict):
        serialized = json.dumps(arr, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(arr, pd.DataFrame):
        serialized = arr.to_json(orient="split", double_precision=10).encode("utf-8")
    elif isinstance(arr, pd.Series):
        serialized = arr.to_json(orient="split", double_precision=10).encode("utf-8")
    elif isinstance(arr, (np.ndarray, list, tuple)):
        np_arr = np.asarray(arr, dtype=float)
        serialized = np_arr.tobytes()
    else:
        serialized = str(arr).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:32]


# =========================================================================== #
# 1. SHOCK NORMALIZATION & COMPATIBILITY VALIDATION
# =========================================================================== #


def normalize_shock(raw_value: float, shock_unit: ShockUnit | str) -> tuple[float, str, str]:
    """Normalize a raw financial shock into its canonical computational unit.

    Conventions:
    - RETURN_DECIMAL: -0.10 means -10% return (identity, computational unit = RETURN_DECIMAL).
    - RELATIVE_PERCENT: -10.0 means -10% (raw / 100.0 -> -0.10, computational unit = RETURN_DECIMAL).
    - BASIS_POINTS: +100.0 means +0.0100 absolute change (raw / 10000.0, computational unit = ABSOLUTE).
    - VOLATILITY_POINTS: +5.0 means +5 vol percentage points (raw / 100.0 -> +0.0500, computational unit = ABSOLUTE).
    - ABSOLUTE: identity (computational unit = ABSOLUTE).
    - LOG_RETURN: exp(raw) - 1.0 (computational unit = RETURN_DECIMAL).
    """
    unit_str = shock_unit.value if isinstance(shock_unit, ShockUnit) else str(shock_unit).upper()
    val = float(raw_value)

    if unit_str == ShockUnit.RETURN_DECIMAL.value:
        return val, "RETURN_DECIMAL: identity", ShockUnit.RETURN_DECIMAL.value
    elif unit_str == ShockUnit.RELATIVE_PERCENT.value:
        return val / 100.0, "RELATIVE_PERCENT: raw / 100.0 -> RETURN_DECIMAL", ShockUnit.RETURN_DECIMAL.value
    elif unit_str == ShockUnit.BASIS_POINTS.value:
        return val / 10000.0, "BASIS_POINTS: raw / 10000.0 -> ABSOLUTE", ShockUnit.ABSOLUTE.value
    elif unit_str == ShockUnit.VOLATILITY_POINTS.value:
        return val / 100.0, "VOLATILITY_POINTS: raw / 100.0 -> ABSOLUTE", ShockUnit.ABSOLUTE.value
    elif unit_str == ShockUnit.ABSOLUTE.value:
        return val, "ABSOLUTE: identity", ShockUnit.ABSOLUTE.value
    elif unit_str == ShockUnit.LOG_RETURN.value:
        return (
            math.exp(val) - 1.0,
            "LOG_RETURN: exp(raw) - 1.0 -> RETURN_DECIMAL",
            ShockUnit.RETURN_DECIMAL.value,
        )
    else:
        raise ValueError(f"Unsupported ShockUnit: {shock_unit!r}")


def create_scenario_shock(
    risk_factor_id: str,
    raw_value: float,
    shock_unit: ShockUnit | str = ShockUnit.RETURN_DECIMAL,
    shock_space: ShockSpace | str = ShockSpace.ASSET_RETURN,
    base_value: float | None = None,
    currency: str | None = None,
    source_reference: str | None = None,
) -> ScenarioShock:
    """Construct an explicit ScenarioShock with deterministic normalization."""
    norm_val, norm_rule, comp_unit = normalize_shock(raw_value, shock_unit)
    return ScenarioShock(
        risk_factor_id=str(risk_factor_id),
        shock_space=shock_space.value if isinstance(shock_space, ShockSpace) else str(shock_space),
        shock_unit=shock_unit.value if isinstance(shock_unit, ShockUnit) else str(shock_unit),
        raw_value=float(raw_value),
        normalized_value=norm_val,
        normalization_rule=norm_rule,
        computational_unit=comp_unit,
        base_value=base_value,
        currency=currency,
        source_reference=source_reference,
    )


def validate_repricing_shock_compatibility(
    method: RepricingMethod | str,
    shock_space: ShockSpace | str,
) -> tuple[bool, str]:
    """Validate mathematical compatibility between repricing method and shock space."""
    m_str = method.value if isinstance(method, RepricingMethod) else str(method).upper()
    s_str = shock_space.value if isinstance(shock_space, ShockSpace) else str(shock_space).upper()

    if m_str == RepricingMethod.LINEAR_RETURN.value:
        if s_str in (ShockSpace.ASSET_RETURN.value, ShockSpace.PRICE.value):
            return True, "Compatible: Linear return repricing on asset returns/prices."
        return (
            False,
            f"Incompatible: Method LINEAR_RETURN does not support shock space {s_str}. Requires ASSET_RETURN.",
        )

    elif m_str == RepricingMethod.FACTOR_LINEAR.value:
        if s_str == ShockSpace.FACTOR_RETURN.value:
            return True, "Compatible: Factor linear repricing on factor returns."
        return False, f"Incompatible: Method FACTOR_LINEAR requires FACTOR_RETURN shock space, got {s_str}."

    elif m_str in (RepricingMethod.DELTA.value, RepricingMethod.DELTA_GAMMA.value):
        # Delta/Gamma methods support any risk factor space provided sensitivities are explicitly defined
        return (
            True,
            f"Compatible: Sensitivity-based repricing ({m_str}) supports {s_str} with aligned sensitivities.",
        )

    elif m_str == RepricingMethod.FULL_REVALUATION_ADAPTER.value:
        return (
            False,
            "FULL_REVALUATION_ADAPTER unavailable: instrument pricing adapters are deferred in Gate 6.",
        )

    elif m_str == RepricingMethod.CUSTOM_DETERMINISTIC_ADAPTER.value:
        return True, "Compatible: Custom deterministic adapter."

    return False, f"Unknown repricing method: {method!r}"


# =========================================================================== #
# 2. DATA INTEGRITY CHECKER
# =========================================================================== #


def validate_scenario_data_integrity(
    spec: ScenarioSpec,
    assets: Sequence[str] | None = None,
    factors: Sequence[str] | None = None,
    sensitivities: dict[str, SensitivitySpec] | None = None,
    portfolio_assets: Sequence[str] | None = None,
    portfolio_weights: dict[str, float] | None = None,
) -> ScenarioDataIntegrityResult:
    """Deterministically audit a scenario specification for semantic correctness, coverage, and provenance."""
    if assets is None:
        if portfolio_assets is not None:
            assets = portfolio_assets
        elif portfolio_weights is not None:
            assets = list(portfolio_weights.keys())

    issues: list[str] = []
    shock_spaces: set[str] = set()
    shock_units: set[str] = set()
    shocked_factors: set[str] = set()

    if not spec.scenario_id:
        issues.append("Missing scenario_id.")
    if not spec.shocks:
        issues.append("Scenario contains zero shocks.")

    method_str = (
        spec.repricing_method.value
        if isinstance(spec.repricing_method, RepricingMethod)
        else str(spec.repricing_method)
    )
    type_str = (
        spec.scenario_type.value if isinstance(spec.scenario_type, ScenarioType) else str(spec.scenario_type)
    )

    # 1. Audit per-leg shocks
    for i, s in enumerate(spec.shocks):
        if not s.risk_factor_id:
            issues.append(f"Shock leg {i} missing risk_factor_id.")
        if s.risk_factor_id in shocked_factors:
            issues.append(f"Duplicate shock definition for risk factor '{s.risk_factor_id}'.")
        shocked_factors.add(s.risk_factor_id)

        if not math.isfinite(s.raw_value) or not math.isfinite(s.normalized_value):
            issues.append(
                f"Non-finite shock value in leg '{s.risk_factor_id}': raw={s.raw_value}, norm={s.normalized_value}."
            )

        s_space = s.shock_space.value if isinstance(s.shock_space, ShockSpace) else str(s.shock_space)
        s_unit = s.shock_unit.value if isinstance(s.shock_unit, ShockUnit) else str(s.shock_unit)
        shock_spaces.add(s_space)
        shock_units.add(s_unit)

        # Validate method x shock compatibility
        is_compat, msg = validate_repricing_shock_compatibility(method_str, s_space)
        if not is_compat:
            issues.append(f"Shock leg '{s.risk_factor_id}' ({s_space}): {msg}")

    # 2. Historical Replay Provenance Audit
    if type_str == ScenarioType.HISTORICAL_REPLAY.value:
        if not spec.source_reference:
            issues.append("HISTORICAL_REPLAY scenario requires explicit source_reference.")
        if not spec.as_of_date and not spec.source_fingerprint:
            issues.append("HISTORICAL_REPLAY scenario requires as_of_date or source_fingerprint.")

    # 3. Sensitivity availability for Delta / Delta-Gamma repricing
    sens_complete = True
    if method_str in (RepricingMethod.DELTA.value, RepricingMethod.DELTA_GAMMA.value):
        if sensitivities is None:
            issues.append(f"Method {method_str} requires explicit sensitivities dictionary.")
            sens_complete = False
        else:
            for rf_id in shocked_factors:
                if rf_id not in sensitivities:
                    issues.append(f"Missing sensitivity specification for shocked factor '{rf_id}'.")
                    sens_complete = False
                elif method_str == RepricingMethod.DELTA_GAMMA.value and sensitivities[rf_id].gamma is None:
                    issues.append(
                        f"Missing gamma sensitivity for factor '{rf_id}' under DELTA_GAMMA repricing."
                    )
                    sens_complete = False

    # 4. Asset / Factor Coverage Audit
    cov_complete = True
    if method_str == RepricingMethod.LINEAR_RETURN.value and assets is not None:
        missing_assets = set(assets) - shocked_factors
        if missing_assets:
            issues.append(f"Portfolio assets missing from scenario shocks: {sorted(missing_assets)}.")
            cov_complete = False
    elif method_str == RepricingMethod.FACTOR_LINEAR.value and factors is not None:
        missing_factors = set(factors) - shocked_factors
        if missing_factors:
            issues.append(f"Model factors missing from scenario shocks: {sorted(missing_factors)}.")
            cov_complete = False

    valid = len(issues) == 0
    return ScenarioDataIntegrityResult(
        scenario_id=spec.scenario_id,
        valid=valid,
        n_shocks=len(spec.shocks),
        shock_spaces_present=tuple(sorted(shock_spaces)),
        shock_units_present=tuple(sorted(shock_units)),
        repricing_compatible=all("Incompatible" not in iss for iss in issues),
        sensitivities_complete=sens_complete,
        coverage_complete=cov_complete,
        provenance_valid=type_str != ScenarioType.HISTORICAL_REPLAY.value
        or (bool(spec.source_reference) and (bool(spec.as_of_date) or bool(spec.source_fingerprint))),
        issues=tuple(issues),
        data_fingerprint=_series_fingerprint(
            {"id": spec.scenario_id, "shocks": len(spec.shocks), "valid": valid}
        ),
    )


# =========================================================================== #
# 3. DETERMINISTIC REPRICING ENGINES
# =========================================================================== #


def apply_asset_return_scenario(
    weights: dict[str, float] | pd.Series | np.ndarray,
    scenario_spec_or_shocks: ScenarioSpec | dict[str, float] | Sequence[ScenarioShock],
    portfolio_value: float | None = None,
    assets: Sequence[str] | None = None,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    scenario_id: str | None = None,
) -> ScenarioResult:
    """Evaluate portfolio return, monetary P&L, and asset contributions under asset return shocks.

    Mathematical formulation:
        R_s = sum_i w_i r_{i,s}
        C_i = w_i r_{i,s}
        reconciliation: sum_i C_i == R_s
        canonical scenario_loss = -R_s
    """
    # 1. Normalize portfolio weights
    if isinstance(weights, dict):
        asset_list = list(weights.keys())
        w_vec = np.array([float(weights[a]) for a in asset_list], dtype=float)
    elif isinstance(weights, pd.Series):
        asset_list = list(weights.index)
        w_vec = weights.to_numpy(dtype=float)
    elif isinstance(weights, np.ndarray):
        w_vec = weights.astype(float)
        asset_list = list(assets) if assets is not None else [f"A{i}" for i in range(len(w_vec))]
    else:
        raise TypeError(f"Unsupported weights type: {type(weights)}")

    # 2. Extract normalized shock vector
    scen_id = scenario_id or "SCEN-ASSET"
    scenario_type = ScenarioType.SYNTHETIC.value
    shock_dict: dict[str, float] = {}

    if isinstance(scenario_spec_or_shocks, ScenarioSpec):
        scen_id = scenario_spec_or_shocks.scenario_id
        scenario_type = (
            scenario_spec_or_shocks.scenario_type.value
            if isinstance(scenario_spec_or_shocks.scenario_type, ScenarioType)
            else str(scenario_spec_or_shocks.scenario_type)
        )
        for s in scenario_spec_or_shocks.shocks:
            shock_dict[s.risk_factor_id] = s.normalized_value
    elif isinstance(scenario_spec_or_shocks, dict):
        shock_dict = {str(k): float(v) for k, v in scenario_spec_or_shocks.items()}
    elif isinstance(scenario_spec_or_shocks, (list, tuple)):
        for s in scenario_spec_or_shocks:
            if isinstance(s, ScenarioShock):
                shock_dict[s.risk_factor_id] = s.normalized_value
            elif isinstance(s, tuple) and len(s) == 2:
                shock_dict[str(s[0])] = float(s[1])

    scenario_id = scen_id

    # Validate asset coverage
    r_vec = np.zeros(len(asset_list), dtype=float)
    missing_assets = []
    for i, a in enumerate(asset_list):
        if a in shock_dict:
            r_vec[i] = shock_dict[a]
        else:
            missing_assets.append(a)

    if missing_assets:
        raise ValueError(f"Scenario shocks missing for portfolio assets: {missing_assets}")

    # 3. Calculate portfolio scenario return & contributions
    contrib_arr = w_vec * r_vec
    port_ret = float(np.sum(contrib_arr))
    port_loss = -port_ret  # Canonical loss sign (positive loss magnitude)

    pnl_val = (portfolio_value * port_ret) if portfolio_value is not None else None
    monetary_loss = -pnl_val if pnl_val is not None else None

    contrib_dict = {a: float(contrib_arr[i]) for i, a in enumerate(asset_list)}
    recon_err = abs(float(np.sum(contrib_arr) - port_ret))

    effective_horizon = horizon
    if isinstance(scenario_spec_or_shocks, ScenarioSpec) and scenario_spec_or_shocks.horizon:
        effective_horizon = scenario_spec_or_shocks.horizon

    limitations = (
        "Linear asset-return repricing: R_s = w' r_s.",
        f"Horizon: {effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon}.",
        "Canonical loss is defined as -Return.",
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        scenario_return=port_ret,
        scenario_loss=port_loss,
        portfolio_value=portfolio_value,
        scenario_pnl=pnl_val,
        scenario_monetary_loss=monetary_loss,
        asset_contributions=contrib_dict,
        factor_contributions={},
        specific_contribution=None,
        group_contributions={},
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        reconciliation_error=recon_err,
        converged=True,
        limitations=limitations,
        data_fingerprint=_series_fingerprint(contrib_arr),
        horizon=str(
            effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon
        ),
        currency=str(scenario_spec_or_shocks.currency)
        if isinstance(scenario_spec_or_shocks, ScenarioSpec)
        else "",
        portfolio_state_fingerprint=_series_fingerprint(
            {a: float(w_vec[i]) for i, a in enumerate(asset_list)}
        ),
    )


def apply_factor_scenario(
    weights: dict[str, float] | pd.Series | np.ndarray,
    exposures: pd.DataFrame | np.ndarray,
    scenario_spec_or_shocks: ScenarioSpec | dict[str, float] | Sequence[ScenarioShock],
    specific_shocks: dict[str, float] | None = None,
    specific_shock_policy: str = "NONE",  # NONE | EXPLICIT_ZERO | SUPPLIED
    portfolio_value: float | None = None,
    assets: Sequence[str] | None = None,
    factors: Sequence[str] | None = None,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    scenario_id: str | None = None,
) -> ScenarioResult:
    """Evaluate portfolio return under factor return shocks and optional specific shocks.

    Mathematical formulation:
        b_p = B' w   (portfolio factor exposures)
        R_factor,s = b_p' f_s
        C_k = b_{p,k} f_{s,k}
        If specific shocks epsilon_s supplied: R_s = R_factor,s + w' epsilon_s
        canonical scenario_loss = -R_s
    """
    # 1. Normalize portfolio weights
    if isinstance(weights, dict):
        asset_list = list(weights.keys())
        w_vec = np.array([float(weights[a]) for a in asset_list], dtype=float)
    elif isinstance(weights, pd.Series):
        asset_list = list(weights.index)
        w_vec = weights.to_numpy(dtype=float)
    elif isinstance(weights, np.ndarray):
        w_vec = weights.astype(float)
        asset_list = list(assets) if assets is not None else [f"A{i}" for i in range(len(w_vec))]
    else:
        raise TypeError(f"Unsupported weights type: {type(weights)}")

    # 2. Normalize exposure matrix B (N x K)
    if isinstance(exposures, pd.DataFrame):
        factor_list = list(exposures.columns)
        B_mat = exposures.loc[asset_list].to_numpy(dtype=float)
    elif isinstance(exposures, np.ndarray):
        B_mat = exposures.astype(float)
        factor_list = list(factors) if factors is not None else [f"F{k}" for k in range(B_mat.shape[1])]
    else:
        raise TypeError(f"Unsupported exposures type: {type(exposures)}")

    # Portfolio factor exposure: b_p = B' w (length K)
    b_p = B_mat.T @ w_vec

    # 3. Extract factor shocks f_s
    scen_id = scenario_id or "SCEN-FACTOR"
    scenario_type = ScenarioType.SYNTHETIC.value
    shock_dict: dict[str, float] = {}

    if isinstance(scenario_spec_or_shocks, ScenarioSpec):
        scen_id = scenario_spec_or_shocks.scenario_id
        scenario_type = (
            scenario_spec_or_shocks.scenario_type.value
            if isinstance(scenario_spec_or_shocks.scenario_type, ScenarioType)
            else str(scenario_spec_or_shocks.scenario_type)
        )
        for s in scenario_spec_or_shocks.shocks:
            shock_dict[s.risk_factor_id] = s.normalized_value
    elif isinstance(scenario_spec_or_shocks, dict):
        shock_dict = {str(k): float(v) for k, v in scenario_spec_or_shocks.items()}
    elif isinstance(scenario_spec_or_shocks, (list, tuple)):
        for s in scenario_spec_or_shocks:
            if isinstance(s, ScenarioShock):
                shock_dict[s.risk_factor_id] = s.normalized_value
            elif isinstance(s, tuple) and len(s) == 2:
                shock_dict[str(s[0])] = float(s[1])

    scenario_id = scen_id

    f_vec = np.zeros(len(factor_list), dtype=float)
    missing_factors = []
    for k, f in enumerate(factor_list):
        if f in shock_dict:
            f_vec[k] = shock_dict[f]
        else:
            missing_factors.append(f)

    if missing_factors:
        raise ValueError(f"Scenario shocks missing for model factors: {missing_factors}")

    # Factor return contributions: C_k = b_{p,k} * f_{s,k}
    factor_contrib_arr = b_p * f_vec
    factor_ret = float(np.sum(factor_contrib_arr))
    factor_contrib_dict = {f: float(factor_contrib_arr[k]) for k, f in enumerate(factor_list)}

    # 4. Handle specific shocks with explicit policy
    spec_policy = specific_shock_policy
    if spec_policy is None and isinstance(scenario_spec_or_shocks, ScenarioSpec):
        spec_policy = scenario_spec_or_shocks.specific_shock_policy
    if spec_policy is None:
        spec_policy = "SUPPLIED" if specific_shocks is not None else "NONE"

    spec_policy = str(spec_policy).upper()
    spec_ret = 0.0
    spec_contrib_val: float | None = None

    if spec_policy == "NONE":
        spec_ret = 0.0
        spec_contrib_val = None
        asset_total_ret = B_mat @ f_vec
    elif spec_policy == "EXPLICIT_ZERO":
        spec_ret = 0.0
        spec_contrib_val = 0.0
        asset_total_ret = B_mat @ f_vec
    elif spec_policy in ("SUPPLIED", "REQUIRED"):
        if specific_shocks is None:
            raise ValueError(
                f"specific_shock_policy is '{spec_policy}' but specific_shocks dictionary is missing/None."
            )
        eps_vec = np.array([float(specific_shocks.get(a, 0.0)) for a in asset_list], dtype=float)
        spec_ret = float(np.sum(w_vec * eps_vec))
        spec_contrib_val = spec_ret
        asset_total_ret = (B_mat @ f_vec) + eps_vec
    else:
        raise ValueError(
            f"Invalid specific_shock_policy: {spec_policy!r}. Must be 'NONE', 'EXPLICIT_ZERO', or 'SUPPLIED'."
        )

    total_ret = factor_ret + spec_ret
    port_loss = -total_ret

    asset_contrib_arr = w_vec * asset_total_ret
    asset_contrib_dict = {a: float(asset_contrib_arr[i]) for i, a in enumerate(asset_list)}

    pnl_val = (portfolio_value * total_ret) if portfolio_value is not None else None
    monetary_loss = -pnl_val if pnl_val is not None else None
    recon_err = abs(float(np.sum(factor_contrib_arr) + spec_ret - total_ret))

    effective_horizon = horizon
    if isinstance(scenario_spec_or_shocks, ScenarioSpec) and scenario_spec_or_shocks.horizon:
        effective_horizon = scenario_spec_or_shocks.horizon

    limitations = (
        "Factor linear repricing: R_s = (B'w)' f_s + w' epsilon_s.",
        f"Specific shock policy: {spec_policy}.",
        f"Horizon: {effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon}.",
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        repricing_method=RepricingMethod.FACTOR_LINEAR.value,
        scenario_return=total_ret,
        scenario_loss=port_loss,
        portfolio_value=portfolio_value,
        scenario_pnl=pnl_val,
        scenario_monetary_loss=monetary_loss,
        asset_contributions=asset_contrib_dict,
        factor_contributions=factor_contrib_dict,
        specific_contribution=spec_contrib_val,
        group_contributions={},
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        reconciliation_error=recon_err,
        converged=True,
        limitations=limitations,
        data_fingerprint=_series_fingerprint(factor_contrib_arr),
        horizon=str(
            effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon
        ),
        currency=str(scenario_spec_or_shocks.currency)
        if isinstance(scenario_spec_or_shocks, ScenarioSpec)
        else "",
        portfolio_state_fingerprint=_series_fingerprint(
            {a: float(w_vec[i]) for i, a in enumerate(asset_list)}
        ),
    )


def apply_delta_gamma_scenario(
    sensitivities: dict[str, SensitivitySpec] | Sequence[SensitivitySpec],
    scenario_spec_or_shocks: ScenarioSpec | dict[str, float] | Sequence[ScenarioShock],
    gamma_matrix: np.ndarray | pd.DataFrame | list[list[float]] | None = None,
    portfolio_value: float | None = None,
    method: RepricingMethod | str = RepricingMethod.DELTA_GAMMA,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
) -> ScenarioResult:
    """Evaluate second-order sensitivity approximation P&L under risk factor shocks.

    Mathematical formulation:
        P&L_approx = delta' Delta_x + 0.5 Delta_x' Gamma Delta_x
        where Gamma is the full symmetric Hessian matrix.
        If Gamma matrix is omitted but scalar gammas are supplied in SensitivitySpec,
        Gamma is constructed as diag(gamma_i).
    """
    method_str = method.value if isinstance(method, RepricingMethod) else str(method).upper()

    # 1. Parse sensitivities
    if isinstance(sensitivities, (list, tuple)):
        sens_dict = {s.risk_factor_id: s for s in sensitivities}
    elif isinstance(sensitivities, dict):
        sens_dict = dict(sensitivities)
    else:
        raise TypeError(f"Unsupported sensitivities type: {type(sensitivities)}")

    rf_list = sorted(sens_dict.keys())
    delta_vec = np.array([float(sens_dict[rf].delta) for rf in rf_list], dtype=float)

    # 2. Parse Gamma matrix
    if method_str == RepricingMethod.DELTA.value:
        Gamma_mat = np.zeros((len(rf_list), len(rf_list)), dtype=float)
    else:
        if gamma_matrix is not None:
            if isinstance(gamma_matrix, pd.DataFrame):
                Gamma_mat = gamma_matrix.loc[rf_list, rf_list].to_numpy(dtype=float)
            else:
                Gamma_mat = np.asarray(gamma_matrix, dtype=float)
            # Verify Gamma matrix symmetry
            if not np.allclose(Gamma_mat, Gamma_mat.T, atol=1e-8):
                raise ValueError("Gamma matrix must be symmetric (Gamma == Gamma.T).")
        else:
            # Diagonal gamma vector
            gamma_diag = np.array([float(sens_dict[rf].gamma) for rf in rf_list], dtype=float)
            Gamma_mat = np.diag(gamma_diag)

    # 3. Extract normalized shocks
    scenario_id = "SCEN-DELTA-GAMMA"
    scenario_type = ScenarioType.SYNTHETIC.value
    shock_dict: dict[str, float] = {}

    if isinstance(scenario_spec_or_shocks, ScenarioSpec):
        scenario_id = scenario_spec_or_shocks.scenario_id
        scenario_type = (
            scenario_spec_or_shocks.scenario_type.value
            if isinstance(scenario_spec_or_shocks.scenario_type, ScenarioType)
            else str(scenario_spec_or_shocks.scenario_type)
        )
        for s in scenario_spec_or_shocks.shocks:
            shock_dict[s.risk_factor_id] = s.normalized_value
    elif isinstance(scenario_spec_or_shocks, dict):
        shock_dict = {str(k): float(v) for k, v in scenario_spec_or_shocks.items()}
    elif isinstance(scenario_spec_or_shocks, (list, tuple)):
        for s in scenario_spec_or_shocks:
            if isinstance(s, ScenarioShock):
                shock_dict[s.risk_factor_id] = s.normalized_value
            elif isinstance(s, tuple) and len(s) == 2:
                shock_dict[str(s[0])] = float(s[1])

    dx_vec = np.zeros(len(rf_list), dtype=float)
    missing_factors = []
    for i, rf in enumerate(rf_list):
        if rf in shock_dict:
            dx_vec[i] = shock_dict[rf]
        else:
            missing_factors.append(rf)

    if missing_factors:
        raise ValueError(f"Scenario shocks missing for risk factors: {missing_factors}")

    # 4. Compute Delta and Gamma P&L components
    delta_pnl_arr = delta_vec * dx_vec
    delta_pnl = float(np.sum(delta_pnl_arr))

    gamma_pnl_arr = 0.5 * dx_vec * (Gamma_mat @ dx_vec)
    gamma_pnl = float(np.sum(gamma_pnl_arr))

    total_pnl = delta_pnl + gamma_pnl
    monetary_loss = -total_pnl

    port_ret = (total_pnl / portfolio_value) if portfolio_value is not None else total_pnl
    port_loss = -port_ret

    factor_contrib_dict = {rf: float(delta_pnl_arr[i] + gamma_pnl_arr[i]) for i, rf in enumerate(rf_list)}
    recon_err = abs(float(np.sum(delta_pnl_arr) + gamma_pnl - total_pnl))

    effective_horizon = horizon
    if isinstance(scenario_spec_or_shocks, ScenarioSpec) and scenario_spec_or_shocks.horizon:
        effective_horizon = scenario_spec_or_shocks.horizon

    limitations = (
        f"Sensitivity repricing: {method_str}.",
        "P&L_approx = delta' Delta_x + 0.5 Delta_x' Gamma Delta_x.",
        f"Horizon: {effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon}.",
        "Taylor expansion approximation; does not replace full instrument revaluation.",
    )

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        repricing_method=method_str,
        scenario_return=port_ret,
        scenario_loss=port_loss,
        portfolio_value=portfolio_value,
        scenario_pnl=total_pnl if portfolio_value is not None else None,
        scenario_monetary_loss=monetary_loss if portfolio_value is not None else None,
        asset_contributions={},
        factor_contributions=factor_contrib_dict,
        specific_contribution=None,
        group_contributions={},
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        reconciliation_error=recon_err,
        converged=True,
        limitations=limitations,
        data_fingerprint=_series_fingerprint(dx_vec),
        horizon=str(
            effective_horizon.value if isinstance(effective_horizon, MetricHorizon) else effective_horizon
        ),
        currency=str(scenario_spec_or_shocks.currency)
        if isinstance(scenario_spec_or_shocks, ScenarioSpec)
        else "",
        portfolio_state_fingerprint=_series_fingerprint(
            {rf: (s.delta, s.gamma) for rf, s in sens_dict.items()}
        ),
    )


def apply_benchmark_active_scenario(
    weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    benchmark_weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    scenario_spec_or_shocks: ScenarioSpec | dict[str, float] | Sequence[ScenarioShock] | None = None,
    exposures: pd.DataFrame | np.ndarray | None = None,
    assets: Sequence[str] | None = None,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    portfolio_weights: dict[str, float] | pd.Series | np.ndarray | None = None,
) -> ActiveScenarioResult:
    """Evaluate scenario returns for portfolio, benchmark, and active portfolio.

    Mathematical identity:
        R_port = w' r_s
        R_bmk = w_b' r_s
        R_active = (w - w_b)' r_s
        identity: R_port - R_bmk == R_active
    """
    if weights is None and portfolio_weights is not None:
        weights = portfolio_weights
    if weights is None:
        raise ValueError("weights (or portfolio_weights) must be provided.")
    if benchmark_weights is None:
        raise ValueError("benchmark_weights must be provided.")
    if scenario_spec_or_shocks is None:
        raise ValueError("scenario_spec_or_shocks must be provided.")
    if isinstance(weights, dict):
        all_assets = sorted(
            set(weights.keys()) | set(benchmark_weights.keys() if isinstance(benchmark_weights, dict) else ())
        )
        w_vec = np.array([float(weights.get(a, 0.0)) for a in all_assets], dtype=float)
    elif isinstance(weights, pd.Series):
        all_assets = sorted(
            set(weights.index)
            | set(benchmark_weights.index if isinstance(benchmark_weights, pd.Series) else ())
        )
        w_vec = np.array([float(weights.get(a, 0.0)) for a in all_assets], dtype=float)
    else:
        w_vec = np.asarray(weights, dtype=float)
        all_assets = list(assets) if assets is not None else [f"A{i}" for i in range(len(w_vec))]

    if isinstance(benchmark_weights, (dict, pd.Series)):
        wb_vec = np.array([float(benchmark_weights.get(a, 0.0)) for a in all_assets], dtype=float)
    else:
        wb_vec = np.asarray(benchmark_weights, dtype=float)

    # Active weights: w_act = w - w_b
    w_act = w_vec - wb_vec

    # 1. Asset return repricing
    port_res = apply_asset_return_scenario(
        weights=dict(zip(all_assets, w_vec, strict=True)),
        scenario_spec_or_shocks=scenario_spec_or_shocks,
        horizon=horizon,
    )
    bmk_res = apply_asset_return_scenario(
        weights=dict(zip(all_assets, wb_vec, strict=True)),
        scenario_spec_or_shocks=scenario_spec_or_shocks,
        horizon=horizon,
    )
    act_res = apply_asset_return_scenario(
        weights=dict(zip(all_assets, w_act, strict=True)),
        scenario_spec_or_shocks=scenario_spec_or_shocks,
        horizon=horizon,
    )

    recon_err = abs(float(port_res.scenario_return - bmk_res.scenario_return - act_res.scenario_return))

    # Optional factor contributions for active portfolio
    act_factor_dict: dict[str, float] = {}
    if exposures is not None:
        if isinstance(exposures, pd.DataFrame):
            B_mat = exposures.loc[all_assets].to_numpy(dtype=float)
            f_names = list(exposures.columns)
        else:
            B_mat = np.asarray(exposures, dtype=float)
            f_names = [f"F{k}" for k in range(B_mat.shape[1])]
        b_act = B_mat.T @ w_act
        # Extract factor shocks
        if isinstance(scenario_spec_or_shocks, ScenarioSpec):
            f_shocks = {s.risk_factor_id: s.normalized_value for s in scenario_spec_or_shocks.shocks}
            f_vec = np.array([f_shocks.get(f, 0.0) for f in f_names], dtype=float)
            act_factor_dict = {f: float(b_act[k] * f_vec[k]) for k, f in enumerate(f_names)}

    return ActiveScenarioResult(
        scenario_id=port_res.scenario_id,
        portfolio_return=port_res.scenario_return,
        benchmark_return=bmk_res.scenario_return,
        active_return=act_res.scenario_return,
        portfolio_loss=port_res.scenario_loss,
        benchmark_loss=bmk_res.scenario_loss,
        active_loss=act_res.scenario_loss,
        active_asset_contributions=act_res.asset_contributions,
        active_factor_contributions=act_factor_dict,
        reconciliation_error=recon_err,
        data_fingerprint=_series_fingerprint(
            {"p": port_res.scenario_return, "b": bmk_res.scenario_return, "a": act_res.scenario_return}
        ),
    )


apply_active_scenario = apply_benchmark_active_scenario


def apply_group_scenario_decomposition(
    asset_contributions: dict[str, float],
    group_mapping: dict[str, str | list[str]],
    partition_contract: PartitionContract | str = PartitionContract.EXHAUSTIVE_PARTITION,
) -> dict[str, float]:
    """Aggregate asset-level scenario contributions into group/sector stress impacts.

    Under EXHAUSTIVE_PARTITION (disjoint groups), sum_g C_g == total_return.
    Under OVERLAPPING_ANALYTICAL, non-additive group views are reported without forced reconciliation.
    """
    group_totals: dict[str, float] = {}

    for asset, contrib in asset_contributions.items():
        groups = group_mapping.get(asset, "UNMAPPED")
        if isinstance(groups, str):
            groups = [groups]
        elif not isinstance(groups, (list, tuple)):
            groups = ["UNMAPPED"]

        for g in groups:
            group_totals[g] = group_totals.get(g, 0.0) + float(contrib)

    return group_totals


def evaluate_scenario(
    spec: ScenarioSpec,
    weights: dict[str, float] | pd.Series | np.ndarray,
    exposures: pd.DataFrame | np.ndarray | None = None,
    sensitivities: dict[str, SensitivitySpec] | Sequence[SensitivitySpec] | None = None,
    portfolio_value: float | None = None,
    assets: Sequence[str] | None = None,
    factors: Sequence[str] | None = None,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
) -> ScenarioResult:
    """Production orchestration entry point: integrity validation strictly precedes repricing."""
    sens_dict: dict[str, SensitivitySpec] | None = None
    if sensitivities is not None:
        if isinstance(sensitivities, dict):
            sens_dict = sensitivities
        else:
            sens_dict = {s.risk_factor_id: s for s in sensitivities}

    # 1. Pre-flight deterministic integrity validation
    diag = validate_scenario_data_integrity(
        spec=spec,
        assets=assets,
        factors=factors,
        sensitivities=sens_dict,
        portfolio_weights=weights if isinstance(weights, dict) else None,
    )
    if not diag.valid:
        raise ValueError(
            f"Scenario data integrity validation failed for '{spec.scenario_id}' with {len(diag.issues)} issue(s): {list(diag.issues)}"
        )

    # 2. Dispatch to deterministic repricing engine
    method_str = (
        spec.repricing_method.value
        if isinstance(spec.repricing_method, RepricingMethod)
        else str(spec.repricing_method).upper()
    )

    if method_str == RepricingMethod.LINEAR_RETURN.value:
        return apply_asset_return_scenario(
            weights=weights,
            scenario_spec_or_shocks=spec,
            portfolio_value=portfolio_value,
            assets=assets,
            horizon=horizon,
        )
    elif method_str == RepricingMethod.FACTOR_LINEAR.value:
        if exposures is None:
            raise ValueError("FACTOR_LINEAR repricing requires exposures matrix.")
        return apply_factor_scenario(
            weights=weights,
            exposures=exposures,
            scenario_spec_or_shocks=spec,
            portfolio_value=portfolio_value,
            assets=assets,
            factors=factors,
            horizon=horizon,
        )
    elif method_str in (RepricingMethod.DELTA.value, RepricingMethod.DELTA_GAMMA.value):
        if sens_dict is None:
            raise ValueError(f"{method_str} repricing requires sensitivities.")
        return apply_delta_gamma_scenario(
            sensitivities=sens_dict,
            scenario_spec_or_shocks=spec,
            portfolio_value=portfolio_value,
            horizon=horizon,
        )
    else:
        raise NotImplementedError(f"Repricing method '{method_str}' is not supported or deferred.")


# =========================================================================== #
# 4. MULTI-SCENARIO SET COMPARISON & SENSITIVITY GRIDS
# =========================================================================== #


def compare_scenario_set(
    scenario_results: Sequence[ScenarioResult],
    ranking_metric: str = "scenario_loss",
    allow_cross_method: bool = True,
) -> ScenarioSetResult:
    """Deterministically rank and compare a set of heterogeneous scenarios by loss.

    Adverse ranking convention: descending canonical scenario_loss (worst scenario is maximum loss).
    """
    if not scenario_results:
        raise ValueError("Cannot compare an empty set of scenario results.")

    valid_metrics = ("scenario_loss", "scenario_return", "scenario_pnl", "scenario_monetary_loss")
    if ranking_metric not in valid_metrics:
        raise ValueError(f"Unsupported ranking_metric '{ranking_metric}'. Supported metrics: {valid_metrics}")

    scen_ids = tuple(r.scenario_id for r in scenario_results)
    ret_dict = {r.scenario_id: r.scenario_return for r in scenario_results}
    loss_dict = {r.scenario_id: r.scenario_loss for r in scenario_results}
    pnl_dict = {
        r.scenario_id: (r.scenario_pnl if r.scenario_pnl is not None else 0.0) for r in scenario_results
    }
    method_dict = {r.scenario_id: r.repricing_method for r in scenario_results}

    # Sort descending by scenario_loss (worst loss first)
    ranked_ids = tuple(sorted(scen_ids, key=lambda sid: loss_dict[sid], reverse=True))

    worst_id = ranked_ids[0]
    best_id = ranked_ids[-1]

    methods_present = set(method_dict.values())
    limitations: list[str] = [
        f"Multi-scenario set ranked descending by '{ranking_metric}'.",
        f"Worst scenario: '{worst_id}' with loss {loss_dict[worst_id]:.4f}.",
    ]

    incompatibilities: list[str] = []

    # 1. Horizon Compatibility
    horizons_present = {r.horizon for r in scenario_results if r.horizon}
    if len(horizons_present) > 1:
        incompatibilities.append(f"Incompatible scenario horizons: {sorted(horizons_present)}")

    # 2. Base Currency Compatibility
    currencies_present = {r.currency for r in scenario_results if r.currency}
    if len(currencies_present) > 1:
        incompatibilities.append(f"Incompatible scenario base currencies: {sorted(currencies_present)}")

    # 3. Portfolio State Compatibility
    state_fps_present = {
        r.portfolio_state_fingerprint for r in scenario_results if r.portfolio_state_fingerprint
    }
    if len(state_fps_present) > 1:
        incompatibilities.append("Incompatible portfolio states across scenarios in comparison set")

    # 4. Loss / P&L Basis Compatibility
    if ranking_metric in ("scenario_pnl", "scenario_monetary_loss"):
        pv_set = {r.portfolio_value for r in scenario_results if r.portfolio_value is not None}
        has_none_pv = any(r.portfolio_value is None for r in scenario_results)
        if has_none_pv:
            incompatibilities.append("Monetary P&L ranking requires portfolio_value for all scenarios")
        elif len(pv_set) > 1:
            incompatibilities.append(
                f"Incompatible portfolio values for monetary comparison: {sorted(pv_set)}"
            )

    # 5. Repricing Method Compatibility
    if len(methods_present) > 1:
        if allow_cross_method:
            limitations.append(
                f"Cross-method comparison: methods present {sorted(methods_present)}. Method differences disclosed."
            )
        else:
            incompatibilities.append(
                f"Incompatible repricing methods present in scenario set: {sorted(methods_present)}"
            )

    comparability_valid = len(incompatibilities) == 0
    if not comparability_valid:
        limitations.extend(incompatibilities)

    return ScenarioSetResult(
        scenarios_evaluated=scen_ids,
        ranking_metric=ranking_metric,
        scenario_returns=ret_dict,
        scenario_losses=loss_dict,
        scenario_pnls=pnl_dict,
        loss_rankings=ranked_ids,
        worst_scenario_id=worst_id,
        best_scenario_id=best_id,
        worst_scenario_loss=loss_dict[worst_id],
        best_scenario_loss=loss_dict[best_id],
        method_disclosures=method_dict,
        comparability_valid=comparability_valid,
        limitations=tuple(limitations),
        data_fingerprint=_series_fingerprint(loss_dict),
    )


def evaluate_scenario_sensitivity_grid(
    base_spec: ScenarioSpec,
    risk_factor_id: str,
    shock_multipliers: Sequence[float],
    weights: dict[str, float] | pd.Series | np.ndarray,
    portfolio_value: float | None = None,
) -> ScenarioSensitivityResult:
    """Evaluate response curves across a deterministic parameter sweep for a selected risk factor."""
    base_shock_val: float | None = None
    shock_unit = ShockUnit.RETURN_DECIMAL
    shock_space = ShockSpace.ASSET_RETURN

    for s in base_spec.shocks:
        if s.risk_factor_id == risk_factor_id:
            base_shock_val = s.raw_value
            shock_unit = s.shock_unit if isinstance(s.shock_unit, ShockUnit) else ShockUnit(s.shock_unit)
            shock_space = (
                s.shock_space if isinstance(s.shock_space, ShockSpace) else ShockSpace(s.shock_space)
            )
            break

    if base_shock_val is None:
        raise ValueError(f"Risk factor '{risk_factor_id}' not found in base scenario spec.")

    grid_points: list[ScenarioSensitivityPoint] = []
    losses: list[float] = []

    for m in shock_multipliers:
        raw_m = base_shock_val * float(m)
        norm_m, _, _ = normalize_shock(raw_m, shock_unit)

        # Build modified spec
        mod_shocks = [
            s
            if s.risk_factor_id != risk_factor_id
            else create_scenario_shock(risk_factor_id, raw_m, shock_unit, shock_space)
            for s in base_spec.shocks
        ]
        mod_spec = ScenarioSpec(
            scenario_id=f"{base_spec.scenario_id}_m_{m:.2f}",
            scenario_name=f"{base_spec.scenario_name} (x{m:.2f})",
            scenario_type=base_spec.scenario_type,
            shocks=tuple(mod_shocks),
            repricing_method=base_spec.repricing_method,
            horizon=base_spec.horizon,
        )

        res = apply_asset_return_scenario(
            weights=weights, scenario_spec_or_shocks=mod_spec, portfolio_value=portfolio_value
        )
        grid_points.append(
            ScenarioSensitivityPoint(
                shock_multiplier=float(m),
                raw_shock_value=raw_m,
                normalized_shock_value=norm_m,
                portfolio_return=res.scenario_return,
                portfolio_loss=res.scenario_loss,
                portfolio_pnl=res.scenario_pnl,
            )
        )
        losses.append(res.scenario_loss)

    base_idx = shock_multipliers.index(1.0) if 1.0 in shock_multipliers else 0
    base_loss = losses[base_idx]

    return ScenarioSensitivityResult(
        risk_factor_id=risk_factor_id,
        grid_points=tuple(grid_points),
        base_loss=base_loss,
        max_loss=max(losses),
        min_loss=min(losses),
        data_fingerprint=_series_fingerprint(losses),
    )


# =========================================================================== #
# 5. REVERSE STRESS TESTING ENGINE
# =========================================================================== #


def solve_reverse_stress(
    spec: ReverseStressSpec,
    sensitivities_or_weights: dict[str, float | SensitivitySpec] | pd.Series | np.ndarray,
    factors: Sequence[str] | None = None,
) -> ReverseStressResult:
    """Solve for the minimum risk factor shock vector achieving a target portfolio loss.

    Mathematical formulation:
        min_x ||x||_M
        constraint: -c' x >= target_loss  <=>  c' x <= -target_loss
    """
    if spec.target_loss <= 0:
        raise ValueError(
            f"ReverseStressSpec.target_loss must be strictly positive (> 0), got {spec.target_loss}."
        )

    repricing_str = (
        spec.repricing_method.value
        if isinstance(spec.repricing_method, RepricingMethod)
        else str(spec.repricing_method).upper()
    )
    if repricing_str in (RepricingMethod.DELTA_GAMMA.value, "DELTA_GAMMA", "FULL_REVALUATION_ADAPTER"):
        raise NotImplementedError(
            f"{repricing_str} reverse stress is deferred in Gate 6. Supported methods: LINEAR_RETURN, FACTOR_LINEAR."
        )

    # 1. Parse linear sensitivity vector c
    if isinstance(sensitivities_or_weights, dict):
        rf_list = list(sensitivities_or_weights.keys())
        c_vals: list[float] = []
        for rf in rf_list:
            item = sensitivities_or_weights[rf]
            if isinstance(item, SensitivitySpec):
                c_vals.append(float(item.delta))
            else:
                c_vals.append(float(item))
        c_vec = np.array(c_vals, dtype=np.float64)
    elif isinstance(sensitivities_or_weights, pd.Series):
        rf_list = list(sensitivities_or_weights.index)
        c_vec = sensitivities_or_weights.to_numpy(dtype=np.float64)
    elif isinstance(sensitivities_or_weights, np.ndarray):
        c_vec = sensitivities_or_weights.astype(np.float64)
        rf_list = list(factors) if factors is not None else [f"F{k}" for k in range(len(c_vec))]
    else:
        raise TypeError(f"Unsupported sensitivities type: {type(sensitivities_or_weights)}")

    K = len(c_vec)
    target_L = float(spec.target_loss)
    norm_str = (
        spec.distance_norm.value
        if isinstance(spec.distance_norm, ReverseStressNorm)
        else str(spec.distance_norm).upper()
    )

    # Verify non-zero linear sensitivity
    c_norm_sq = float(np.sum(np.square(c_vec)))
    if c_norm_sq < 1e-12:
        raise ValueError(
            f"All sensitivities in reverse stress problem are zero ({c_norm_sq:.2e}); target loss cannot be achieved."
        )

    # Heterogeneous coordinate check for unscaled L2
    if norm_str == ReverseStressNorm.L2.value:
        if getattr(spec, "is_heterogeneous_unscaled", False):
            raise ValueError(
                "Unscaled L2 distance is undefined across heterogeneous financial coordinates. Provide scaling_factors (WEIGHTED_L2) or reference_covariance (MAHALANOBIS)."
            )
        if getattr(spec, "risk_factor_units", None) and len(set(spec.risk_factor_units.values())) > 1:
            raise ValueError(
                "Unscaled L2 distance is undefined across heterogeneous financial coordinates. Provide scaling_factors (WEIGHTED_L2) or reference_covariance (MAHALANOBIS)."
            )

    # 2. Case A: Unconstrained L2 Closed Form
    active_bounds = dict(spec.shock_bounds)
    if hasattr(spec, "bounds") and spec.bounds:
        active_bounds.update(spec.bounds)
    has_bounds = bool(active_bounds)
    is_unconstrained_l2 = norm_str == ReverseStressNorm.L2.value and not has_bounds

    if is_unconstrained_l2:
        # Minimum L2 shock satisfying -c' x >= target_L (active equality -c' x = target_L):
        # x* = -target_L * c / (c' c)
        x_star = (-target_L / c_norm_sq) * c_vec
        achieved_ret = float(np.sum(c_vec * x_star))
        achieved_loss = -achieved_ret
        dist = float(np.sqrt(np.sum(np.square(x_star))))

        shock_dict = {rf: float(x_star[i]) for i, rf in enumerate(rf_list)}
        return ReverseStressResult(
            target_loss=target_L,
            achieved_loss=achieved_loss,
            achieved_return=achieved_ret,
            loss_gap=abs(achieved_loss - target_L),
            shock_vector=shock_dict,
            normalized_shocks=shock_dict,
            distance=dist,
            distance_norm=norm_str,
            bounds_satisfied=True,
            solver_status="OPTIMAL_CLOSED_FORM",
            converged=True,
            is_closed_form=True,
            limitations=("Analytical minimum L2 reverse stress solution (unconstrained).",),
            data_fingerprint=_series_fingerprint(x_star),
        )

    # 3. Case B: Numerical Optimization (Bounded, Weighted L2, or Mahalanobis)
    if norm_str == ReverseStressNorm.MAHALANOBIS.value:
        active_cov = (
            spec.reference_covariance
            if spec.reference_covariance is not None
            else getattr(spec, "covariance", None)
        )
        if active_cov is None:
            raise ValueError("Mahalanobis reverse stress requires reference_covariance.")
        if isinstance(active_cov, pd.DataFrame):
            if list(active_cov.index) != rf_list or list(active_cov.columns) != rf_list:
                raise ValueError("Reference covariance labels or order do not match factor list.")
        Sigma_mat = np.asarray(active_cov, dtype=float)
        if Sigma_mat.shape != (K, K):
            raise ValueError(f"Reference covariance shape {Sigma_mat.shape} does not match factor count {K}.")
        # Unit compatibility check for Mahalanobis geometry
        cov_unit = spec.provenance.get("covariance_unit") if spec.provenance else None
        if cov_unit and spec.risk_factor_units:
            for rf, unit in spec.risk_factor_units.items():
                if unit != cov_unit:
                    raise ValueError(
                        f"Mahalanobis geometry unit mismatch: reference covariance is in '{cov_unit}' but factor '{rf}' is in '{unit}'."
                    )
        if (
            spec.risk_factor_units
            and len(set(spec.risk_factor_units.values())) > 1
            and not spec.scaling_factors
        ):
            raise ValueError(
                "Mahalanobis geometry requires homogeneous factor coordinates or explicit scaling_factors across heterogeneous units."
            )

        evals = np.linalg.eigvalsh(Sigma_mat)
        if np.any(evals <= 1e-12):
            raise ValueError(
                "Mahalanobis reference covariance must be non-singular and strictly positive-definite (found non-positive eigenvalue)."
            )
        try:
            Sigma_inv = np.linalg.inv(Sigma_mat)
        except np.linalg.LinAlgError as err:
            raise ValueError(f"Mahalanobis reference covariance inversion failed: {err}") from err

        def objective(x: np.ndarray) -> float:
            return float(x.T @ Sigma_inv @ x)

        def grad_obj(x: np.ndarray) -> np.ndarray:
            return 2.0 * Sigma_inv @ x

    elif norm_str == ReverseStressNorm.WEIGHTED_L2.value:
        active_weights = getattr(spec, "weight_matrix", None)
        if active_weights is not None:
            if isinstance(active_weights, pd.DataFrame):
                if list(active_weights.index) != rf_list or list(active_weights.columns) != rf_list:
                    raise ValueError("Weighted L2 metric matrix labels or order do not match factor list.")
            if isinstance(active_weights, np.ndarray) and active_weights.ndim == 1:
                W_diag = np.diag(active_weights.astype(float))
            elif isinstance(active_weights, np.ndarray) and active_weights.ndim == 2:
                W_diag = active_weights.astype(float)
            else:
                W_diag = np.diag(np.asarray(active_weights, dtype=float))
            if W_diag.shape != (K, K):
                raise ValueError(f"Weight matrix shape {W_diag.shape} does not match factor count {K}.")
            evals = np.linalg.eigvalsh(W_diag)
            if np.any(evals <= 1e-12):
                raise ValueError("Weighted L2 metric matrix must be strictly positive-definite.")
        else:
            w_scales = np.array([float(spec.scaling_factors.get(rf, 1.0)) for rf in rf_list], dtype=float)
            if np.any(w_scales <= 0):
                raise ValueError("Weighted L2 norm requires strictly positive scale weights.")
            W_diag = np.diag(w_scales)

        def objective(x: np.ndarray) -> float:
            return float(x.T @ W_diag @ x)

        def grad_obj(x: np.ndarray) -> np.ndarray:
            return 2.0 * W_diag @ x

    else:  # Bounded L2

        def objective(x: np.ndarray) -> float:
            return float(np.sum(np.square(x)))

        def grad_obj(x: np.ndarray) -> np.ndarray:
            return 2.0 * x

    # Constraint: -c' x >= target_L  <=>  -target_L - c' x >= 0
    cons = [{"type": "ineq", "fun": lambda x: -target_L - float(np.sum(c_vec * x)), "jac": lambda x: -c_vec}]

    # Bounds
    bounds_list: list[tuple[float | None, float | None]] = []
    for rf in rf_list:
        if rf in active_bounds:
            b_low, b_high = active_bounds[rf]
            bounds_list.append((b_low, b_high))
        else:
            bounds_list.append((None, None))

    # Initial guess: unconstrained L2 direction
    x0 = (-target_L / c_norm_sq) * c_vec

    res = optimize.minimize(
        objective,
        x0=x0,
        jac=grad_obj,
        bounds=bounds_list,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )

    achieved_ret = float(np.sum(c_vec * res.x))
    achieved_loss = -achieved_ret
    loss_gap = abs(achieved_loss - target_L)

    # Post-solve verification: target loss achieved and bounds satisfied
    target_achieved = achieved_loss >= target_L - 1e-6
    bounds_ok = True
    for i, _rf in enumerate(rf_list):
        b_low_val, b_high_val = bounds_list[i]
        if b_low_val is not None and res.x[i] < float(b_low_val) - 1e-6:
            bounds_ok = False
        if b_high_val is not None and res.x[i] > float(b_high_val) + 1e-6:
            bounds_ok = False

    converged = bool(res.success and target_achieved and bounds_ok)
    solver_st = "OPTIMAL" if converged else ("INFEASIBLE" if not target_achieved else "BOUNDS_BREACHED")

    if norm_str == ReverseStressNorm.MAHALANOBIS.value:
        dist = float(np.sqrt(max(0.0, float(res.x.T @ Sigma_inv @ res.x))))
    elif norm_str == ReverseStressNorm.WEIGHTED_L2.value:
        dist = float(np.sqrt(max(0.0, float(res.x.T @ W_diag @ res.x))))
    else:
        dist = float(np.sqrt(max(0.0, float(np.sum(np.square(res.x))))))

    shock_dict = {rf: float(res.x[i]) for i, rf in enumerate(rf_list)}
    limitations = (
        f"Reverse stress under {norm_str} norm.",
        "Minimum shock under selected geometry; does NOT imply most likely market crisis.",
        f"Target loss = {target_L:.4f}, Achieved loss = {achieved_loss:.4f}.",
    )

    return ReverseStressResult(
        target_loss=target_L,
        achieved_loss=achieved_loss,
        achieved_return=achieved_ret,
        loss_gap=loss_gap,
        shock_vector=shock_dict,
        normalized_shocks=shock_dict,
        distance=dist,
        distance_norm=norm_str,
        bounds_satisfied=bounds_ok,
        solver_status=solver_st,
        converged=converged,
        is_closed_form=False,
        limitations=limitations,
        data_fingerprint=_series_fingerprint(res.x),
    )


# =========================================================================== #
# 6. HISTORICAL REPLAY ENGINE
# =========================================================================== #


def replay_historical_scenario(
    historical_shocks: dict[str, float] | pd.Series,
    portfolio_weights: dict[str, float] | pd.Series,
    source_reference: str,
    observation_date: str,
    proxy_mappings: dict[str, str] | None = None,
    portfolio_value: float | None = None,
    source_fingerprint: str = "",
    source_currency: str | None = None,
    portfolio_currency: str | None = None,
    fx_policy: dict[str, Any] | str | None = None,
) -> ScenarioResult:
    """Replay historical return observations across portfolio assets with explicit provenance.

    Missing asset policy: fail closed unless explicit proxy mapping is supplied.
    Currency policy: fail closed on currency mismatch without explicit FX conversion policy.
    """
    if not source_reference:
        raise ValueError("Historical replay requires explicit source_reference.")
    if not observation_date:
        raise ValueError("Historical replay requires explicit observation_date.")
    if (
        source_currency
        and portfolio_currency
        and source_currency.upper() != portfolio_currency.upper()
        and not fx_policy
    ):
        raise ValueError(
            f"Currency mismatch between historical scenario source ({source_currency}) and portfolio base ({portfolio_currency}) "
            f"without explicit FX conversion policy."
        )

    if isinstance(portfolio_weights, pd.Series):
        weights_dict = portfolio_weights.to_dict()
    else:
        weights_dict = dict(portfolio_weights)

    if isinstance(historical_shocks, pd.Series):
        hist_dict = historical_shocks.to_dict()
    else:
        hist_dict = dict(historical_shocks)

    resolved_shocks: list[ScenarioShock] = []
    missing_assets: list[str] = []
    has_mapping = False

    for asset, _w in weights_dict.items():
        if asset in hist_dict:
            raw_r = float(hist_dict[asset])
            resolved_shocks.append(
                create_scenario_shock(
                    risk_factor_id=asset,
                    raw_value=raw_r,
                    shock_unit=ShockUnit.RETURN_DECIMAL,
                    shock_space=ShockSpace.ASSET_RETURN,
                    source_reference=source_reference,
                )
            )
        elif proxy_mappings and asset in proxy_mappings:
            proxy_asset = proxy_mappings[asset]
            if proxy_asset in hist_dict:
                raw_r = float(hist_dict[proxy_asset])
                has_mapping = True
                resolved_shocks.append(
                    ScenarioShock(
                        risk_factor_id=asset,
                        shock_space=ShockSpace.ASSET_RETURN.value,
                        shock_unit=ShockUnit.RETURN_DECIMAL.value,
                        raw_value=raw_r,
                        normalized_value=raw_r,
                        normalization_rule="RETURN_DECIMAL: identity",
                        source_reference=source_reference,
                        provenance={"proxy_source": proxy_asset, "mapping_type": "EXPLICIT_PROXY"},
                    )
                )
            else:
                missing_assets.append(f"{asset} (proxy '{proxy_asset}' missing from historical shocks)")
        else:
            missing_assets.append(asset)

    if missing_assets:
        raise ValueError(f"Historical shocks missing for assets (no valid proxy): {missing_assets}")

    scenario_type = (
        ScenarioType.HISTORICAL_REPLAY.value if not has_mapping else "HISTORICAL_REPLAY_WITH_MAPPING"
    )

    spec = ScenarioSpec(
        scenario_id=f"HIST-{observation_date}",
        scenario_name=f"Historical Replay ({observation_date})",
        scenario_type=scenario_type,
        shocks=tuple(resolved_shocks),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        as_of_date=observation_date,
        source_reference=source_reference,
        source_fingerprint=source_fingerprint or _series_fingerprint(hist_dict),
    )

    return apply_asset_return_scenario(
        weights=weights_dict,
        scenario_spec_or_shocks=spec,
        portfolio_value=portfolio_value,
    )
