"""Analytical Manifest and Workbench Lens Specifications for StART.

Defines the declarative contracts for all supported model families and techniques:
- Required vs conditional stages
- Semantic data roles
- Required vs conditional vs not-applicable analytical artifacts
- Metric families and diagnostics
- Comparison compatibility
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticalArtifactRequirement:
    """Artifact specification rule with strict applicability semantics."""

    artifact_type: str
    label: str
    status: str  # "REQUIRED" | "CONDITIONAL" | "NOT_APPLICABLE"
    description: str


@dataclass(frozen=True)
class AnalyticalManifest:
    """Declarative scientific manifest for a model family and technique."""

    model_family: str
    technique: str
    task_type: str
    label: str
    data_roles: tuple[str, ...]
    required_stages: tuple[str, ...]
    metric_families: tuple[str, ...]
    diagnostic_families: tuple[str, ...]
    structural_analyses: tuple[str, ...]
    sensitivity_dimensions: tuple[str, ...]
    artifact_requirements: tuple[AnalyticalArtifactRequirement, ...]
    comparison_compatible_with: tuple[str, ...]

    def get_required_artifact_types(self) -> list[str]:
        return [r.artifact_type for r in self.artifact_requirements if r.status == "REQUIRED"]

    def get_not_applicable_types(self) -> list[str]:
        return [r.artifact_type for r in self.artifact_requirements if r.status == "NOT_APPLICABLE"]


# --------------------------------------------------------------------------- #
# Manifest Registry for Acceptance Matrix Cases A through I
# --------------------------------------------------------------------------- #

MANIFEST_REGISTRY: dict[str, AnalyticalManifest] = {
    # Case A: Predictive Classification
    "predictive_classification": AnalyticalManifest(
        model_family="predictive_ml",
        technique="logistic_regression",
        task_type="binary_classification",
        label="Supervised Binary Classification (Tabular)",
        data_roles=("features", "target", "sample_weight"),
        required_stages=(
            "data_selection",
            "data_validation",
            "split_protocol",
            "model_fit",
            "performance_evaluation",
            "diagnostics",
            "explainability",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("discrimination", "threshold_metrics", "calibration", "loss"),
        diagnostic_families=("confusion_matrix", "roc_curve", "pr_curve", "calibration_curve"),
        structural_analyses=("permutation_importance",),
        sensitivity_dimensions=("regularization_strength", "decision_threshold"),
        artifact_requirements=(
            AnalyticalArtifactRequirement("data_profile_table", "Data Profile Table", "REQUIRED", "Dataset cardinality and missingness"),
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Configuration", "REQUIRED", "Model parameters and split specification"),
            AnalyticalArtifactRequirement("metric_summary", "Performance Metric Summary", "REQUIRED", "Scalar discrimination and calibration metrics"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix Table", "REQUIRED", "Thresholded classification breakdown"),
            AnalyticalArtifactRequirement("roc_curve", "ROC Discrimination Curve", "REQUIRED", "FPR vs TPR curve"),
            AnalyticalArtifactRequirement("calibration_curve", "Reliability Diagram", "REQUIRED", "ECE decile reliability plot"),
            AnalyticalArtifactRequirement("feature_importance", "Global Feature Importance", "REQUIRED", "Permutation feature importance rankings"),
            AnalyticalArtifactRequirement("sensitivity_summary", "Sensitivity Summary Table", "REQUIRED", "Regularization and threshold stability deltas"),
            AnalyticalArtifactRequirement("baseline_comparison", "Baseline Comparison", "REQUIRED", "Evaluation against majority class baseline"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Tree clustering is not applicable to tabular classification"),
            AnalyticalArtifactRequirement("ndcg_ranking_table", "Top-K Ranking Table", "NOT_APPLICABLE", "Ranking cutoffs not applicable to binary classification"),
        ),
        comparison_compatible_with=("logistic_regression", "gradient_boosting", "torch_mlp"),
    ),

    # Case B: Predictive Regression
    "predictive_regression": AnalyticalManifest(
        model_family="predictive_ml",
        technique="ridge_regression",
        task_type="regression",
        label="Supervised Continuous Regression (Tabular)",
        data_roles=("features", "target"),
        required_stages=(
            "data_selection",
            "data_validation",
            "split_protocol",
            "model_fit",
            "performance_evaluation",
            "residual_diagnostics",
            "explainability",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("error_metrics", "explained_variance"),
        diagnostic_families=("residual_distribution", "error_quantiles", "predicted_vs_actual"),
        structural_analyses=("feature_coefficients",),
        sensitivity_dimensions=("regularization_alpha",),
        artifact_requirements=(
            AnalyticalArtifactRequirement("data_profile_table", "Data Profile Table", "REQUIRED", "Dataset cardinality and continuous target stats"),
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Configuration", "REQUIRED", "Model parameters and split specification"),
            AnalyticalArtifactRequirement("metric_summary", "Regression Performance Table", "REQUIRED", "RMSE, MAE, R², MSE metrics"),
            AnalyticalArtifactRequirement("residual_diagnostics", "Residual Diagnostics Table", "REQUIRED", "Residual error quantiles and skewness"),
            AnalyticalArtifactRequirement("predicted_vs_actual", "Predicted vs Actual Plot", "REQUIRED", "Prediction fidelity series"),
            AnalyticalArtifactRequirement("feature_importance", "Feature Coefficients", "REQUIRED", "Standardized regression coefficients"),
            AnalyticalArtifactRequirement("sensitivity_summary", "Sensitivity Summary Table", "REQUIRED", "Alpha perturbation impact on RMSE"),
            AnalyticalArtifactRequirement("baseline_comparison", "Baseline Comparison", "REQUIRED", "Evaluation against mean target baseline"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix", "NOT_APPLICABLE", "Discrete confusion matrix not applicable to regression"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Tree clustering not applicable to regression"),
        ),
        comparison_compatible_with=("ridge_regression", "linear_regression", "random_forest_regressor"),
    ),

    # Case C: Tiny Deep Learning Classification
    "deep_learning": AnalyticalManifest(
        model_family="deep_learning",
        technique="torch_mlp",
        task_type="binary_classification",
        label="Deep Neural Network (PyTorch MLP)",
        data_roles=("features", "target"),
        required_stages=(
            "data_selection",
            "data_validation",
            "architecture_summary",
            "bounded_training_loop",
            "training_curves",
            "performance_evaluation",
            "diagnostics",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("accuracy", "cross_entropy_loss", "discrimination"),
        diagnostic_families=("training_loss_history", "validation_loss_history", "checkpoint_summary"),
        structural_analyses=("architecture_layers", "permutation_importance"),
        sensitivity_dimensions=("learning_rate", "random_seed"),
        artifact_requirements=(
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved DL Configuration", "REQUIRED", "Layer dims, lr, dropout, batch_size, epochs"),
            AnalyticalArtifactRequirement("architecture_summary", "Neural Architecture Summary", "REQUIRED", "Layer specifications and parameter count"),
            AnalyticalArtifactRequirement("training_curve", "Training & Validation Loss Curve", "REQUIRED", "Epoch-by-epoch loss convergence data"),
            AnalyticalArtifactRequirement("checkpoint_summary", "Best Checkpoint Summary", "REQUIRED", "Best epoch and minimum validation loss"),
            AnalyticalArtifactRequirement("metric_summary", "Out-of-Sample Performance Table", "REQUIRED", "Holdout accuracy, precision, recall, F1, loss"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix Table", "REQUIRED", "Holdout binary classification counts"),
            AnalyticalArtifactRequirement("sensitivity_summary", "Sensitivity & Stability Table", "REQUIRED", "Learning rate perturbation metrics"),
            AnalyticalArtifactRequirement("baseline_comparison", "Baseline Comparison", "REQUIRED", "Performance relative to linear classifier"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Clustering not applicable to DL classifier"),
        ),
        comparison_compatible_with=("torch_mlp", "logistic_regression"),
    ),

    # Case D: Matrix Factorization Recommender
    "recommender_mf": AnalyticalManifest(
        model_family="recommender",
        technique="matrix_factorization",
        task_type="rating_prediction",
        label="Biased Matrix Factorization (Explicit Ratings)",
        data_roles=("user_id", "item_id", "rating"),
        required_stages=(
            "data_selection",
            "interaction_profiling",
            "user_stratified_split",
            "factor_optimization",
            "rating_evaluation",
            "ranking_evaluation",
            "cold_start_cohorts",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("rating_fidelity", "top_k_ranking"),
        diagnostic_families=("matrix_sparsity", "interaction_distribution"),
        structural_analyses=("cold_start_degradation",),
        sensitivity_dimensions=("latent_dimension", "regularization"),
        artifact_requirements=(
            AnalyticalArtifactRequirement("interaction_data_profile", "Matrix Sparsity & Density", "REQUIRED", "Cardinality, sparsity ratio, fill rate"),
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Configuration", "REQUIRED", "Latent factors, learning rate, epochs, regularization"),
            AnalyticalArtifactRequirement("rating_metric_table", "Rating Prediction Metrics", "REQUIRED", "RMSE, MAE, R² on test holdout"),
            AnalyticalArtifactRequirement("ranking_metric_table", "Top-K Ranking Performance", "REQUIRED", "NDCG@K, Recall@K, Precision@K cutoffs"),
            AnalyticalArtifactRequirement("cold_start_table", "Cold-Start Cohort Analysis", "REQUIRED", "Warm vs cold user degradation metrics"),
            AnalyticalArtifactRequirement("sensitivity_table", "Latent Sensitivity Analysis", "REQUIRED", "Perturbation grid with delta RMSE"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Portfolio tree clustering not applicable to recommender"),
        ),
        comparison_compatible_with=("matrix_factorization", "neural_collaborative_filtering"),
    ),

    # Case E: NCF Recommender
    "recommender_ncf": AnalyticalManifest(
        model_family="recommender",
        technique="neural_collaborative_filtering",
        task_type="top_k_ranking",
        label="Neural Collaborative Filtering (Implicit Ranking)",
        data_roles=("user_id", "item_id", "interaction"),
        required_stages=(
            "data_selection",
            "interaction_profiling",
            "negative_sampling",
            "user_stratified_split",
            "neural_training_loop",
            "ranking_evaluation",
            "beyond_accuracy_evaluation",
            "cold_start_cohorts",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("top_k_ranking", "beyond_accuracy"),
        diagnostic_families=("matrix_sparsity", "popularity_distribution"),
        structural_analyses=("catalog_coverage", "novelty_bits", "popularity_bias"),
        sensitivity_dimensions=("embedding_dimension", "dropout"),
        artifact_requirements=(
            AnalyticalArtifactRequirement("interaction_data_profile", "Matrix Sparsity & Density", "REQUIRED", "Cardinality, sparsity ratio, fill rate"),
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Configuration", "REQUIRED", "Embedding dims, MLP layers, lr, epochs"),
            AnalyticalArtifactRequirement("ranking_metric_table", "Top-K Ranking Performance", "REQUIRED", "NDCG@K, Recall@K, Precision@K, MRR, HitRate@K"),
            AnalyticalArtifactRequirement("beyond_accuracy_table", "Coverage & Diversity Table", "REQUIRED", "Catalog coverage, user coverage, novelty, popularity bias"),
            AnalyticalArtifactRequirement("cold_start_table", "Cold-Start Cohort Analysis", "REQUIRED", "Warm vs cold user degradation metrics"),
            AnalyticalArtifactRequirement("sensitivity_table", "Embedding Sensitivity Analysis", "REQUIRED", "Perturbation grid with delta NDCG@10"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Dendrogram not applicable to NCF"),
        ),
        comparison_compatible_with=("neural_collaborative_filtering", "matrix_factorization"),
    ),

    # Case F: Factorization Machine Recommender
    "recommender_fm": AnalyticalManifest(
        model_family="recommender",
        technique="factorization_machine",
        task_type="contextual_ranking",
        label="2-Way Factorization Machine (Contextual Interactions)",
        data_roles=("user_id", "item_id", "user_features", "item_features", "context_features", "target"),
        required_stages=(
            "data_selection",
            "feature_encoding",
            "pairwise_factorization",
            "interaction_scoring",
            "ranking_evaluation",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("discrimination", "ranking"),
        diagnostic_families=("feature_contributions", "linear_vs_pairwise"),
        structural_analyses=("pairwise_interactions",),
        sensitivity_dimensions=("latent_dimension", "regularization"),
        artifact_requirements=(
            AnalyticalArtifactRequirement("interaction_data_profile", "Contextual Data Profile", "REQUIRED", "Features, interactions, sparsity"),
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved FM Configuration", "REQUIRED", "Latent factors, learning rate, epochs"),
            AnalyticalArtifactRequirement("ranking_metric_table", "Interaction Ranking Performance", "REQUIRED", "NDCG@K, Recall@K cutoffs"),
            AnalyticalArtifactRequirement("metric_summary", "Interaction Probability Metrics", "REQUIRED", "Logloss, interaction ROC-AUC"),
            AnalyticalArtifactRequirement("sensitivity_table", "Latent Sensitivity Analysis", "REQUIRED", "Perturbation grid with delta AUC"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Dendrogram not applicable to FM"),
        ),
        comparison_compatible_with=("factorization_machine", "logistic_regression"),
    ),

    # Case G: HRP Portfolio
    "portfolio_hrp": AnalyticalManifest(
        model_family="portfolio",
        technique="hierarchical_risk_parity",
        task_type="portfolio_optimization",
        label="Hierarchical Risk Parity (HRP)",
        data_roles=("asset_returns", "covariance_matrix"),
        required_stages=(
            "universe_selection",
            "covariance_estimation",
            "correlation_distance",
            "hierarchical_clustering",
            "quasi_diagonalization",
            "recursive_bisection",
            "weight_allocation",
            "risk_decomposition",
            "performance_evaluation",
            "linkage_sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("risk_metrics", "performance_metrics", "concentration_metrics"),
        diagnostic_families=("correlation_matrix", "distance_matrix", "linkage_tree", "dendrogram"),
        structural_analyses=("quasi_diagonal_order", "cluster_tree", "component_risk_contributions"),
        sensitivity_dimensions=("linkage_method",),
        artifact_requirements=(
            AnalyticalArtifactRequirement("correlation_matrix", "Correlation Matrix & Heatmap", "REQUIRED", "Pairwise Pearson correlation coefficients"),
            AnalyticalArtifactRequirement("distance_matrix", "Angular Distance Matrix", "REQUIRED", "Continuous angular distance d=sqrt(0.5*(1-rho))"),
            AnalyticalArtifactRequirement("linkage_matrix", "Hierarchical Linkage Matrix", "REQUIRED", "Canonical cluster tree agglomeration schedule"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram (SVG)", "REQUIRED", "Visual tree topology reflecting canonical linkage"),
            AnalyticalArtifactRequirement("asset_weights_table", "Final Portfolio Weights Table", "REQUIRED", "Asset weights summing to 1.0"),
            AnalyticalArtifactRequirement("risk_contribution_table", "Risk Contribution Decomposition", "REQUIRED", "Component & percentage risk contributions"),
            AnalyticalArtifactRequirement("performance_table", "Backtest Performance Metrics", "REQUIRED", "Annualized return, volatility, Sharpe, max drawdown"),
            AnalyticalArtifactRequirement("sensitivity_summary", "Linkage Sensitivity Analysis", "REQUIRED", "Weight delta across linkage methods (single/complete/average)"),
            AnalyticalArtifactRequirement("baseline_comparison", "Equal-Weight Comparison", "REQUIRED", "HRP vs 1/N allocation risk and Sharpe comparison"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix", "NOT_APPLICABLE", "Confusion matrix not applicable to portfolio"),
        ),
        comparison_compatible_with=("hierarchical_risk_parity", "equal_risk_contribution", "minimum_variance", "equal_weight"),
    ),

    # Case H: Minimum Variance Portfolio
    "portfolio_min_variance": AnalyticalManifest(
        model_family="portfolio",
        technique="minimum_variance",
        task_type="portfolio_optimization",
        label="Global Minimum Variance Portfolio",
        data_roles=("asset_returns", "covariance_matrix"),
        required_stages=(
            "universe_selection",
            "covariance_estimation",
            "convex_qp_solve",
            "weight_allocation",
            "risk_decomposition",
            "performance_evaluation",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("portfolio_volatility", "diversification_ratio", "weight_concentration"),
        diagnostic_families=("covariance_condition_number", "constraint_violations"),
        structural_analyses=("component_risk_contributions",),
        sensitivity_dimensions=("covariance_shrinkage",),
        artifact_requirements=(
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Optimizer Configuration", "REQUIRED", "Objective, constraints, solver settings"),
            AnalyticalArtifactRequirement("asset_weights_table", "Minimum Variance Weights", "REQUIRED", "Constrained optimal asset allocation"),
            AnalyticalArtifactRequirement("risk_contribution_table", "Risk Contribution Decomposition", "REQUIRED", "Component risk contributions"),
            AnalyticalArtifactRequirement("performance_table", "Portfolio Performance Metrics", "REQUIRED", "Annualized volatility, return, Sharpe"),
            AnalyticalArtifactRequirement("baseline_comparison", "Equal-Weight Comparison", "REQUIRED", "Variance reduction relative to 1/N"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "Non-hierarchical optimizer does not produce a dendrogram"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix", "NOT_APPLICABLE", "Not applicable to portfolio"),
        ),
        comparison_compatible_with=("minimum_variance", "hierarchical_risk_parity", "equal_weight"),
    ),

    # Case I: Equal Risk Contribution Portfolio
    "portfolio_erc": AnalyticalManifest(
        model_family="portfolio",
        technique="equal_risk_contribution",
        task_type="portfolio_optimization",
        label="Equal Risk Contribution (Risk Parity)",
        data_roles=("asset_returns", "covariance_matrix"),
        required_stages=(
            "universe_selection",
            "covariance_estimation",
            "log_barrier_solve",
            "weight_allocation",
            "risk_budget_reconciliation",
            "performance_evaluation",
            "sensitivity",
            "deterministic_findings",
            "artifact_generation",
        ),
        metric_families=("portfolio_volatility", "risk_dispersion", "target_risk_budget"),
        diagnostic_families=("risk_budget_residuals", "solver_convergence"),
        structural_analyses=("component_risk_contributions", "percentage_risk_contributions"),
        sensitivity_dimensions=("risk_target",),
        artifact_requirements=(
            AnalyticalArtifactRequirement("resolved_configuration", "Resolved Optimizer Configuration", "REQUIRED", "Log-barrier formulation settings"),
            AnalyticalArtifactRequirement("asset_weights_table", "ERC Portfolio Weights Table", "REQUIRED", "Risk parity optimal asset weights"),
            AnalyticalArtifactRequirement("risk_contribution_table", "Risk Contribution Reconciliation", "REQUIRED", "Component risk contributions vs 1/N target"),
            AnalyticalArtifactRequirement("performance_table", "Portfolio Performance Metrics", "REQUIRED", "Annualized volatility, return, Sharpe"),
            AnalyticalArtifactRequirement("baseline_comparison", "Equal-Weight Comparison", "REQUIRED", "ERC vs 1/N risk dispersion comparison"),
            AnalyticalArtifactRequirement("dendrogram", "HRP Dendrogram", "NOT_APPLICABLE", "ERC does not produce a hierarchical dendrogram"),
            AnalyticalArtifactRequirement("confusion_matrix", "Confusion Matrix", "NOT_APPLICABLE", "Not applicable to portfolio"),
        ),
        comparison_compatible_with=("equal_risk_contribution", "hierarchical_risk_parity", "minimum_variance", "equal_weight"),
    ),
}


def get_manifest(case_or_technique: str) -> AnalyticalManifest:
    """Resolve the analytical manifest by case key or technique name."""
    case_map = {
        "case_a": "predictive_classification",
        "case_b": "predictive_regression",
        "case_c": "deep_learning_mlp",
        "case_d": "recommender_mf",
        "case_e": "recommender_ncf",
        "case_f": "recommender_fm",
        "case_g": "portfolio_hrp",
        "case_h": "portfolio_min_variance",
        "case_i": "portfolio_erc",
    }
    resolved_key = case_map.get(case_or_technique, case_or_technique)
    if resolved_key in MANIFEST_REGISTRY:
        return MANIFEST_REGISTRY[resolved_key]

    for m in MANIFEST_REGISTRY.values():
        if m.technique == resolved_key or m.model_family == resolved_key:
            return m

    return MANIFEST_REGISTRY["predictive_classification"]
