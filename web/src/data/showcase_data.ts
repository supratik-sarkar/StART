import { ReviewPresentationExport } from "../types/start_schema";

export const MARKET_SHOWCASE: ReviewPresentationExport = {
  run_id: "RUN-SHOWCASE-MARKET-01",
  mode: "deterministic",
  domains: ["market"],
  materiality: "high",
  lifecycle: "validation",
  governance_disposition: "ACCEPT",
  attestation_seal_merkle_root: "9e8a7b6c5d4e3f210987654321abcdef0123456789abcdef0123456789abcdef",
  blocks: {
    PORTFOLIO: {
      block_id: "PORTFOLIO",
      title: "Portfolio Construction & Risk Allocation",
      domain: "market",
      rows: [
        { test_id: "portfolio.hrp_weights", metric: "effective_n_assets", value: 14.8, status: "PASS", evidence_id: "EV-MKT-001" },
        { test_id: "portfolio.herc_clustering", metric: "tree_cophenetic_corr", value: 0.884, status: "PASS", evidence_id: "EV-MKT-002" },
        { test_id: "portfolio.diversification_ratio", metric: "diversification_ratio", value: 2.34, status: "PASS", evidence_id: "EV-MKT-003" },
      ],
      summary: { total_assets: 25, algorithm: "Hierarchical Equal Risk Contribution (HERC)" },
    },
    TAIL_RISK: {
      block_id: "TAIL_RISK",
      title: "Value-at-Risk & Backtesting Diagnostics",
      domain: "market",
      rows: [
        { test_id: "traded_risk.var_kupiec_pof", metric: "empirical_exceptions", value: 3, status: "PASS", evidence_id: "EV-MKT-004" },
        { test_id: "traded_risk.var_kupiec_pof", metric: "p_value", value: 0.421, status: "PASS", evidence_id: "EV-MKT-004" },
        { test_id: "traded_risk.var_christoffersen_independence", metric: "lr_independence", value: 0.82, status: "PASS", evidence_id: "EV-MKT-005" },
      ],
      summary: { confidence_level: 0.99, horizon_days: 1, test_window: 250 },
    },
    SCENARIO_STRESS: {
      block_id: "SCENARIO_STRESS",
      title: "Scenario & Reverse Stress Testing",
      domain: "market",
      rows: [
        { test_id: "stress.2008_lehman_replay", metric: "max_portfolio_drawdown", value: -0.162, unit: "%", status: "PASS", evidence_id: "EV-MKT-006" },
        { test_id: "stress.covid_liquidity_shock", metric: "liquidity_adjusted_shortfall", value: -0.094, unit: "%", status: "PASS", evidence_id: "EV-MKT-007" },
        { test_id: "stress.reverse_stress_breach", metric: "min_shock_to_capital_breach", value: 3.42, unit: "sigma", status: "PASS", evidence_id: "EV-MKT-008" },
      ],
    },
  },
  orchestration_events: [
    { event_id: "EVT-01", event_type: "agent_transition", source_agent: "Director", target_agent: "MarketSpecialist", stage: "PLANNING", action: "discover_applicable_tests", status: "SUCCESS", latency_ms: 12.4 },
    { event_id: "EVT-02", event_type: "tool_execution", source_agent: "MarketSpecialist", target_agent: "DeterministicEngine", stage: "EXECUTION", action: "portfolio.hrp_weights", status: "SUCCESS", latency_ms: 45.1, evidence_refs: ["EV-MKT-001"] },
    { event_id: "EVT-03", event_type: "tool_execution", source_agent: "MarketSpecialist", target_agent: "DeterministicEngine", stage: "EXECUTION", action: "traded_risk.var_kupiec_pof", status: "SUCCESS", latency_ms: 32.8, evidence_refs: ["EV-MKT-004"] },
    { event_id: "EVT-04", event_type: "policy_decision", source_agent: "PolicyEngine", target_agent: "Director", stage: "GOVERNANCE", action: "opa_evaluate_bundle", policy_decision: "ALLOW", status: "SUCCESS", latency_ms: 18.2 },
    { event_id: "EVT-05", event_type: "governance_seal", source_agent: "Director", target_agent: "AttestationLedger", stage: "ATTESTATION", action: "merkle_seal_commit", status: "SUCCESS", latency_ms: 22.0 },
  ],
};

export const PREDICTIVE_SHOWCASE: ReviewPresentationExport = {
  run_id: "RUN-SHOWCASE-PREDICTIVE-01",
  mode: "deterministic",
  domains: ["predictive"],
  materiality: "high",
  lifecycle: "validation",
  governance_disposition: "ACCEPT",
  attestation_seal_merkle_root: "1f2e3d4c5b6a79887766554433221100aabbccddeeff00112233445566778899",
  blocks: {
    DATA_PREPROCESSING: {
      block_id: "DATA_PREPROCESSING",
      title: "Data Quality & Feature Engineering Diagnostics",
      domain: "predictive",
      rows: [
        { test_id: "data.missing_value_rate", metric: "max_missing_col_rate", value: 0.002, status: "PASS", evidence_id: "EV-PRED-001" },
        { test_id: "data.multicollinearity_vif", metric: "max_vif_score", value: 3.84, status: "PASS", evidence_id: "EV-PRED-002" },
        { test_id: "data.target_leakage_detector", metric: "max_feature_target_mutual_info", value: 0.142, status: "PASS", evidence_id: "EV-PRED-003" },
      ],
    },
    PERFORMANCE: {
      block_id: "PERFORMANCE",
      title: "Discriminatory Power & Calibration",
      domain: "predictive",
      rows: [
        { test_id: "metrics.roc_auc", metric: "test_auc", value: 0.842, status: "PASS", evidence_id: "EV-PRED-004" },
        { test_id: "metrics.brier_score", metric: "brier_score", value: 0.089, status: "PASS", evidence_id: "EV-PRED-005" },
        { test_id: "metrics.kolmogorov_smirnov", metric: "ks_statistic", value: 0.548, status: "PASS", evidence_id: "EV-PRED-006" },
      ],
    },
    EXPLAINABILITY: {
      block_id: "EXPLAINABILITY",
      title: "Explainability & Feature Attributions (SHAP)",
      domain: "predictive",
      rows: [
        { test_id: "xai.shap_global_importance", metric: "top1_feature_share", value: 0.284, status: "PASS", evidence_id: "EV-PRED-007" },
        { test_id: "xai.shap_additivity_check", metric: "max_reconstruction_error", value: 1.2e-6, status: "PASS", evidence_id: "EV-PRED-008" },
      ],
    },
  },
  orchestration_events: [
    { event_id: "EVT-P01", event_type: "agent_transition", source_agent: "Director", target_agent: "CreditSpecialist", stage: "DISCOVERY", action: "discover_applicable_tests", status: "SUCCESS", latency_ms: 14.1 },
    { event_id: "EVT-P02", event_type: "tool_execution", source_agent: "CreditSpecialist", target_agent: "DeterministicEngine", stage: "EXECUTION", action: "metrics.roc_auc", status: "SUCCESS", latency_ms: 55.3, evidence_refs: ["EV-PRED-004"] },
    { event_id: "EVT-P03", event_type: "tool_execution", source_agent: "CreditSpecialist", target_agent: "DeterministicEngine", stage: "EXECUTION", action: "xai.shap_global_importance", status: "SUCCESS", latency_ms: 120.4, evidence_refs: ["EV-PRED-007"] },
    { event_id: "EVT-P04", event_type: "policy_decision", source_agent: "PolicyEngine", target_agent: "Director", stage: "GOVERNANCE", action: "opa_evaluate_bundle", policy_decision: "ALLOW", status: "SUCCESS", latency_ms: 16.5 },
  ],
};
