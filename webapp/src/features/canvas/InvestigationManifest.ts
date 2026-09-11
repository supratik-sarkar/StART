export type ModelFamily =
  | "predictive_ml"
  | "deep_learning"
  | "quantitative_finance"
  | "recommender"
  | "scenario";

export interface WorkbenchLens {
  id: string;
  label: string;
  description: string;
  iconName: string;
  badge?: string;
}

export interface ModelWorkbenchManifest {
  family: ModelFamily;
  title: string;
  subtitle: string;
  description: string;
  primaryMetric: string;
  secondaryMetrics: string[];
  lenses: WorkbenchLens[];
  supportedWorkflows: string[];
}

export const RECOMMENDER_LENSES: WorkbenchLens[] = [
  { id: "home", label: "Canvas Home", description: "Executive validation dashboard and attestation overview", iconName: "LayoutDashboard" },
  { id: "interaction_data", label: "Interaction Data", description: "Interaction matrix cardinality, sparsity, and fill rate", iconName: "Grid" },
  { id: "ranking_quality", label: "Ranking Quality", description: "Position-discounted Top-K evaluation (NDCG@K, Recall@K, HitRate@K, MAP@K)", iconName: "Award" },
  { id: "rating_quality", label: "Rating Quality", description: "Explicit rating prediction fidelity (RMSE, MAE, R²)", iconName: "BarChart3" },
  { id: "coverage_diversity", label: "Coverage & Diversity", description: "Catalog coverage, user coverage, novelty bits, and popularity bias", iconName: "Compass" },
  { id: "cold_start", label: "Cold Start", description: "Cold-start cohort degradation analysis (warm vs cold user history)", iconName: "Zap" },
  { id: "sensitivity", label: "Sensitivity", description: "Latent factor dimension and regularization perturbation stability", iconName: "Sliders" },
  { id: "config", label: "Configuration-as-Code", description: "Bitwise-exact executable Python script and JSON execution manifest", iconName: "Code2" },
];

export const PREDICTIVE_ML_LENSES: WorkbenchLens[] = [
  { id: "home", label: "Canvas Home", description: "Predictive model risk validation overview and attestation status", iconName: "LayoutDashboard" },
  { id: "data_diagnostics", label: "Data Diagnostics", description: "Covariate distribution, class imbalance, and data quality", iconName: "Stethoscope" },
  { id: "performance", label: "Discrimination", description: "Holdout discrimination (ROC-AUC, Gini coefficient, KS statistic)", iconName: "BarChart3" },
  { id: "calibration", label: "Calibration", description: "Decile reliability diagram, Expected Calibration Error (ECE), and Brier score", iconName: "Activity" },
  { id: "explainability", label: "Explainability", description: "Permutation feature importance and SHAP-aligned attributions", iconName: "FileCheck" },
  { id: "robustness", label: "Robustness & Sensitivity", description: "Adversarial noise perturbations and stress stability boundaries", iconName: "ShieldCheck" },
  { id: "config", label: "Configuration-as-Code", description: "Executable Python orchestration spec and raw JSON run provenance", iconName: "Code2" },
];

export const DEEP_LEARNING_LENSES: WorkbenchLens[] = [
  { id: "home", label: "Canvas Home", description: "Neural network representation and diagnostic overview", iconName: "LayoutDashboard" },
  { id: "architecture", label: "Architecture", description: "Layer hierarchy, parameter counts, and tensor geometries", iconName: "Cpu" },
  { id: "latent_space", label: "Latent Space", description: "Embedding geometry and hidden activation distributions", iconName: "Layers" },
  { id: "convergence", label: "Convergence", description: "Loss convergence, learning rate dynamics, and gradient norm stability", iconName: "Activity" },
  { id: "calibration", label: "Calibration", description: "Overconfidence screening and probability calibration error", iconName: "BarChart3" },
  { id: "config", label: "Configuration-as-Code", description: "PyTorch execution spec and raw JSON manifest", iconName: "Code2" },
];

export const QUANT_FINANCE_LENSES: WorkbenchLens[] = [
  { id: "home", label: "Canvas Home", description: "Quantitative portfolio risk and traded model validation overview", iconName: "LayoutDashboard" },
  { id: "universe_data", label: "Universe & Covariance", description: "Asset return history, Ledoit-Wolf shrinkage, and matrix condition number", iconName: "Database" },
  { id: "hrp_allocation", label: "HRP Allocation", description: "Hierarchical Risk Parity dendrogram clustering and quasi-diagonalization", iconName: "GitFork" },
  { id: "risk_var", label: "Risk & VaR Backtest", description: "Value-at-Risk Kupiec/Christoffersen backtest and max drawdown", iconName: "ShieldCheck" },
  { id: "factor_exposure", label: "Factor Exposure", description: "Factor loadings and return attribution decomposition", iconName: "Sliders" },
  { id: "config", label: "Configuration-as-Code", description: "Quantitative execution spec and raw JSON manifest", iconName: "Code2" },
];

export const SCENARIO_STRESS_LENSES: WorkbenchLens[] = [
  { id: "home", label: "Canvas Home", description: "Macro stress testing and severe scenario validation overview", iconName: "LayoutDashboard" },
  { id: "scenario_def", label: "Shock Specification", description: "Historical and hypothetical macro factor shocks and shift curves", iconName: "Activity" },
  { id: "loss_distribution", label: "Loss Distribution", description: "Portfolio repricing under severe stress and PnL impact", iconName: "BarChart3" },
  { id: "reverse_stress", label: "Reverse Stress", description: "Search for solvency breach boundaries and capital depletion triggers", iconName: "AlertTriangle" },
  { id: "lineage", label: "Provenance & Lineage", description: "Historical event replay and regulatory scenario lineage", iconName: "Compass" },
  { id: "config", label: "Configuration-as-Code", description: "Scenario execution spec and raw JSON manifest", iconName: "Code2" },
];

export const MODEL_MANIFESTS: Record<ModelFamily, ModelWorkbenchManifest> = {
  recommender: {
    family: "recommender",
    title: "Recommender Systems Validation Workstation",
    subtitle: "Collaborative Filtering, Neural CF & Factorization Machines",
    description:
      "Audit-grade validation vertical for recommender engines. Evaluates position-discounted ranking (NDCG@K, Recall@K), explicit rating RMSE/MAE, beyond-accuracy catalog diversity, cold-start user cohorts, and latent dimension stability.",
    primaryMetric: "NDCG@10",
    secondaryMetrics: ["Recall@10", "Precision@10", "MRR", "Catalog Coverage", "Cold-Start Degradation"],
    lenses: RECOMMENDER_LENSES,
    supportedWorkflows: ["recommender_system"],
  },
  predictive_ml: {
    family: "predictive_ml",
    title: "Predictive ML Validation Workstation",
    subtitle: "Supervised Classification & Tree-Based Models",
    description:
      "Deterministic model risk validation for tabular predictive models. Evaluates holdout discrimination (ROC-AUC, Gini, KS), Brier score calibration, fairness, feature attribution, and adversarial drift.",
    primaryMetric: "ROC-AUC",
    secondaryMetrics: ["Gini", "KS-Statistic", "Brier Score", "ECE", "Feature Drift"],
    lenses: PREDICTIVE_ML_LENSES,
    supportedWorkflows: ["predictive_ml", "data_diagnostics", "model_diagnostics", "calibration", "robustness", "explainability"],
  },
  deep_learning: {
    family: "deep_learning",
    title: "Deep Learning Diagnostic Workstation",
    subtitle: "Neural Latent Representations & Decision Surfaces",
    description:
      "Deep neural network validation with genuine PyTorch computation. Inspects embedding geometry, layer-wise activation distributions, integrated gradients attribution, and calibration error.",
    primaryMetric: "Holdout Accuracy",
    secondaryMetrics: ["Loss Convergence", "Calibration ECE", "Latent Separation", "Gradient Norm"],
    lenses: DEEP_LEARNING_LENSES,
    supportedWorkflows: ["deep_learning"],
  },
  quantitative_finance: {
    family: "quantitative_finance",
    title: "Quantitative Finance & Traded Risk Workstation",
    subtitle: "HRP Allocation, Factor Attribution & Tail Risk",
    description:
      "Quantitative model validation covering Hierarchical Risk Parity (HRP), factor return attributions, Ledoit-Wolf covariance shrinkage, and Kupiec/Christoffersen Value-at-Risk backtests.",
    primaryMetric: "Portfolio Sharpe",
    secondaryMetrics: ["Max Drawdown", "VaR Exceedance Rate", "Condition Number", "Factor Exposure"],
    lenses: QUANT_FINANCE_LENSES,
    supportedWorkflows: ["quantitative_finance"],
  },
  scenario: {
    family: "scenario",
    title: "Scenario & Macro Stress Testing Workstation",
    subtitle: "Historical Replay, Hypothetical Shocks & Reverse Stress",
    description:
      "Capital adequacy and liquidity stress testing validation. Measures portfolio repricing, tail loss distribution, and reverse stress capital exhaustion thresholds under severe macro events.",
    primaryMetric: "Worst-Case PnL",
    secondaryMetrics: ["Loss at 99.9%", "Capital Depletion Ratio", "Reverse Breach Shock", "Tail Vulnerability"],
    lenses: SCENARIO_STRESS_LENSES,
    supportedWorkflows: ["scenario_stress"],
  },
};

export function resolveModelFamily(workflowId?: string, contextId?: string): ModelFamily {
  if (workflowId === "recommender_system" || contextId?.startsWith("recommender_")) {
    return "recommender";
  }
  if (workflowId === "deep_learning" || contextId?.startsWith("deep_learning")) {
    return "deep_learning";
  }
  if (workflowId === "quantitative_finance" || contextId?.startsWith("institutional_market")) {
    return "quantitative_finance";
  }
  if (workflowId === "scenario_stress" || contextId?.startsWith("scenario_")) {
    return "scenario";
  }
  return "predictive_ml";
}
