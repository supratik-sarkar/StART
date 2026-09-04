import React, { useState } from "react";
import {
  Sparkles,
  ArrowRight,
  Cpu,
  Layers,
  BarChart3,
  Sliders,
  Scale,
  Search,
  CheckCircle2,
  ListOrdered,
  Database,
  Activity,
  Shield,
  HelpCircle,
} from "lucide-react";
import { TurnstileWidget } from "./TurnstileWidget";
import { RunRequest } from "../types/start_schema";

export interface WorkflowOption {
  id: string;
  name: string;
  category: "ml_dl" | "quant";
  domain: "predictive" | "deep_learning" | "market";
  syntheticProfile: string;
  description: string;
  defaultPlan: string[];
  defaultArchitecture: string;
  defaultParams: Record<string, any>;
}

export const WORKFLOWS: WorkflowOption[] = [
  {
    id: "predictive_ml",
    name: "Predictive ML",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Evaluate, calibrate and stress-test supervised classification models.",
    defaultArchitecture: "Random Forest & XGBoost",
    defaultPlan: [
      "Data integrity & missing cell diagnostics",
      "Stratified train/test split verification",
      "Baseline classification metrics (ROC-AUC, PR-AUC)",
      "Calibration error assessment (ECE, Brier score)",
      "Robustness against feature noise & perturbations",
      "SHAP feature attribution & explainability",
      "Evidence synthesis & ledger commit",
      "Browser AI qualitative evaluation",
      "Governance disposition & attestation seal",
    ],
    defaultParams: { target: "target", task: "binary_classification", trials: 30 },
  },
  {
    id: "deep_learning",
    name: "Deep Learning",
    category: "ml_dl",
    domain: "deep_learning",
    syntheticProfile: "deep_learning_v1",
    description: "Inspect neural architectures, weight spectra, and gradient norms.",
    defaultArchitecture: "PyTorch MLP & Wide-and-Deep",
    defaultPlan: [
      "Neural architecture & layer parameter inspection",
      "Activation distribution & dead-neuron checks",
      "Loss convergence & learning rate diagnostics",
      "Weight spectra & gradient norm stability",
      "Captum integrated gradients feature attribution",
      "Input perturbation sensitivity analysis",
      "Evidence ledger verification",
      "Concurrent Browser AI interpretation",
      "Merkle root governance attestation",
    ],
    defaultParams: { epochs: 10, batch_size: 32, learning_rate: 0.001 },
  },
  {
    id: "data_diagnostics",
    name: "Data Diagnostics",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Verify data integrity, missingness patterns, and covariate drift.",
    defaultArchitecture: "Statistical Profiler",
    defaultPlan: [
      "Schema & data-type validation",
      "Missingness & sparsity structure analysis",
      "Outlier & distribution anomaly detection",
      "Train/test covariate drift (PSI / KS-test)",
      "Multicollinearity & VIF diagnostics",
      "Evidence commitment to immutable ledger",
      "Governance disposition sign-off",
    ],
    defaultParams: { missing_rate: 0.05, outlier_std: 3.0 },
  },
  {
    id: "model_diagnostics",
    name: "Model Diagnostics",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Deep dive into residual structures, confusion metrics, and error slices.",
    defaultArchitecture: "Gradient Boosted Trees",
    defaultPlan: [
      "Multi-threshold confusion matrix analysis",
      "Subgroup error disparity analysis",
      "Residual autocorrelation & variance diagnostics",
      "High-loss sample cluster isolation",
      "Deterministic metrics grounding",
      "Attestation seal verification",
    ],
    defaultParams: { threshold_steps: 10 },
  },
  {
    id: "calibration",
    name: "Calibration",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Assess probabilistic score reliability, ECE, and reliability curves.",
    defaultArchitecture: "Calibrated Classifier",
    defaultPlan: [
      "Score distribution & probability histogram",
      "Reliability curve (Platt & Isotonic comparison)",
      "Expected Calibration Error (ECE) calculation",
      "Brier score decomposition",
      "Overconfidence penalty assessment",
      "Evidence ledger sign-off",
    ],
    defaultParams: { n_bins: 10 },
  },
  {
    id: "robustness",
    name: "Robustness",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Simulate feature noise, extreme values, and adversarial shifts.",
    defaultArchitecture: "Robust Classifier Suite",
    defaultPlan: [
      "Gaussian feature jitter sweeps (5 levels)",
      "Missing-at-random degradation analysis",
      "Outlier injection stress tests",
      "Worst-case performance boundary estimation",
      "Robustness degradation index computation",
      "Attestation seal commit",
    ],
    defaultParams: { jitter_stds: [0.05, 0.1, 0.2, 0.5] },
  },
  {
    id: "explainability",
    name: "Explainability",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Compute SHAP values, partial dependence, and feature importance.",
    defaultArchitecture: "Tree & Neural Explainers",
    defaultPlan: [
      "Global TreeSHAP / KernelSHAP values",
      "Local feature contribution waterfalls",
      "Partial dependence & ICE curves",
      "Feature interaction detection",
      "Fidelity & explanation stability verification",
      "Evidence ledger provenance commit",
    ],
    defaultParams: { background_samples: 100 },
  },
  {
    id: "hyperparameter_tuning",
    name: "Hyperparameter Tuning",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Bayesian and grid search over hyperparameter manifolds.",
    defaultArchitecture: "Tuned Ensemble",
    defaultPlan: [
      "Parameter search space definition",
      "Stratified k-fold cross validation setup",
      "Trial execution with early stopping (30 trials)",
      "Objective surface & Pareto frontier tracking",
      "Best parameter vector serialization",
      "Out-of-sample confirmation & evidence seal",
    ],
    defaultParams: { strategy: "bayesian", trials: 30, k_folds: 3 },
  },
  {
    id: "model_comparison",
    name: "Model Comparison",
    category: "ml_dl",
    domain: "predictive",
    syntheticProfile: "institutional_credit_v1",
    description: "Benchmark candidate models side-by-side under a shared protocol.",
    defaultArchitecture: "Multi-Candidate Benchmark",
    defaultPlan: [
      "Common evaluation protocol binding",
      "Candidate 1 (Random Forest) evaluation",
      "Candidate 2 (XGBoost) evaluation",
      "Candidate 3 (PyTorch MLP) evaluation",
      "Metric delta & Pareto dominance ranking",
      "Cross-model calibration & robustness comparison",
      "Evidence-linked synthesis & sign-off",
    ],
    defaultParams: { candidates: ["Random Forest", "XGBoost", "PyTorch MLP"] },
  },
  {
    id: "quantitative_finance",
    name: "Quantitative Finance",
    category: "quant",
    domain: "market",
    syntheticProfile: "institutional_market_v1",
    description: "Portfolio risk, covariance matrices, VaR/ES backtesting, and stress testing.",
    defaultArchitecture: "Multi-Asset Factor Risk Model",
    defaultPlan: [
      "Multi-asset price & return series validation",
      "Covariance matrix estimation & regularization",
      "Historical & Parametric Value-at-Risk (VaR 95/99)",
      "Expected Shortfall & Kupiec POF backtests",
      "Factor exposure & risk attribution decomposition",
      "Reverse stress testing & scenario shocks",
      "Evidence ledger commit & Merkle root seal",
    ],
    defaultParams: { var_confidence: 0.99, n_assets: 50, periods: 1000 },
  },
];

interface AgenticComposerProps {
  onLaunchRun: (req: RunRequest) => void;
  isRunning: boolean;
  sessionId: string;
}

export const AgenticComposer: React.FC<AgenticComposerProps> = ({
  onLaunchRun,
  isRunning,
  sessionId,
}) => {
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("predictive_ml");
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [turnstileToken, setTurnstileToken] = useState<string | null>(null);

  // Editable parameters
  const selectedWorkflow =
    WORKFLOWS.find((w) => w.id === selectedWorkflowId) || WORKFLOWS[0];

  const [seed, setSeed] = useState<number>(42);
  const [materiality, setMateriality] = useState<"high" | "medium" | "low">("high");
  const [trialCount, setTrialCount] = useState<number>(30);
  const [activeTab, setActiveTab] = useState<"ALL" | "ML_DL" | "QUANT">("ALL");

  const filteredWorkflows = WORKFLOWS.filter((w) => {
    if (activeTab === "ML_DL") return w.category === "ml_dl";
    if (activeTab === "QUANT") return w.category === "quant";
    return true;
  });

  const handleLaunch = () => {
    const req: RunRequest = {
      domain: selectedWorkflow.domain,
      mode: "deterministic",
      materiality,
      synthetic_profile: selectedWorkflow.syntheticProfile,
      synthetic_profile_version: "1.0.0",
      seed,
      session_id: sessionId,
      workflow: selectedWorkflow.id,
      turnstile_token: turnstileToken,
      parameters: {
        ...selectedWorkflow.defaultParams,
        trials: trialCount,
        prompt: customPrompt,
      },
    };
    onLaunchRun(req);
  };

  return (
    <div className="flex flex-col h-full w-full bg-[#FBFBFA] overflow-y-auto">
      {/* Top Welcome & Intent Composer */}
      <div className="max-w-5xl mx-auto w-full px-8 py-10 flex flex-col gap-8">
        {/* Header Branding */}
        <div className="flex flex-col gap-2 text-left">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-indigo-50 text-indigo-700 border border-indigo-200">
              AGENTIC WORKBENCH
            </span>
            <span className="text-xs text-stone-500 font-mono">StART v4.6.0</span>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-stone-900">
            What do you want to work on?
          </h1>
          <p className="text-sm text-stone-600 max-w-2xl leading-relaxed">
            Select a verified engineering workflow or describe your objective below.
            Deterministic engines compute mathematical proof; AI agents orchestrate and interpret.
          </p>
        </div>

        {/* Intent Description Box */}
        <div className="bg-white border border-[#E5E5E2] rounded-xl p-4 shadow-sm flex flex-col gap-3 focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500 transition-all">
          <div className="flex items-center gap-2 text-xs font-medium text-stone-600">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            <span>Workflow Goal & Intent</span>
          </div>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            placeholder="Describe an analytical workflow, e.g. Evaluate and stress-test a credit classification model under feature noise..."
            rows={2}
            className="w-full text-sm text-stone-900 placeholder:text-stone-400 bg-transparent border-none outline-none resize-none"
          />
        </div>

        {/* Workflow Filter Chips */}
        <div className="flex items-center gap-2 border-b border-[#E5E5E2] pb-3">
          <button
            onClick={() => setActiveTab("ALL")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === "ALL"
                ? "bg-stone-900 text-white"
                : "text-stone-600 hover:bg-stone-100"
            }`}
          >
            All Workflows ({WORKFLOWS.length})
          </button>
          <button
            onClick={() => setActiveTab("ML_DL")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === "ML_DL"
                ? "bg-stone-900 text-white"
                : "text-stone-600 hover:bg-stone-100"
            }`}
          >
            ML & Deep Learning ({WORKFLOWS.filter((w) => w.category === "ml_dl").length})
          </button>
          <button
            onClick={() => setActiveTab("QUANT")}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === "QUANT"
                ? "bg-stone-900 text-white"
                : "text-stone-600 hover:bg-stone-100"
            }`}
          >
            Quantitative Finance ({WORKFLOWS.filter((w) => w.category === "quant").length})
          </button>
        </div>

        {/* Workflow Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {filteredWorkflows.map((w) => {
            const isSelected = w.id === selectedWorkflowId;
            return (
              <div
                key={w.id}
                data-testid={`workflow-card-${w.id}`}
                onClick={() => setSelectedWorkflowId(w.id)}
                className={`flex flex-col p-4 rounded-xl border text-left cursor-pointer transition-all ${
                  isSelected
                    ? "bg-indigo-50/40 border-indigo-600 ring-1 ring-indigo-600 shadow-sm"
                    : "bg-white border-[#E5E5E2] hover:border-stone-400 hover:shadow-sm"
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <span className="font-semibold text-sm text-stone-900">{w.name}</span>
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase font-medium ${
                      w.category === "quant"
                        ? "bg-amber-50 text-amber-700 border border-amber-200"
                        : "bg-indigo-50 text-indigo-700 border border-indigo-200"
                    }`}
                  >
                    {w.domain}
                  </span>
                </div>
                <p className="text-xs text-stone-600 leading-relaxed mb-3 flex-1">
                  {w.description}
                </p>
                <div className="flex items-center justify-between pt-2 border-t border-stone-100 text-[11px] text-stone-500 font-mono">
                  <span>{w.defaultArchitecture}</span>
                  {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" />}
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom Configuration & Pre-Execution Agent Plan */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-4 border-t border-[#E5E5E2]">
          {/* Left 2 Cols: Agent Execution Plan Preview */}
          <div className="lg:col-span-2 bg-white border border-[#E5E5E2] rounded-xl p-5 flex flex-col gap-4 text-left shadow-sm">
            <div className="flex items-center justify-between border-b border-stone-100 pb-3">
              <div className="flex items-center gap-2">
                <ListOrdered className="w-4 h-4 text-indigo-600" />
                <span className="font-semibold text-sm text-stone-900">
                  StART Agent Execution Plan
                </span>
              </div>
              <span className="text-xs text-stone-500 font-mono">
                {selectedWorkflow.defaultPlan.length} deterministic steps
              </span>
            </div>

            <div className="flex flex-col gap-2">
              {selectedWorkflow.defaultPlan.map((step, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-3 py-1 px-2.5 rounded-md hover:bg-stone-50 text-xs text-stone-700 font-mono"
                >
                  <span className="w-5 text-stone-400 font-semibold">{String(idx + 1).padStart(2, "0")}</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-stone-300" />
                  <span className="text-stone-800 flex-1">{step}</span>
                  <span className="text-[10px] text-stone-400">Canonical Engine</span>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-2 text-[11px] text-stone-500 bg-stone-50 p-2.5 rounded border border-stone-200/60 mt-1">
              <Database className="w-3.5 h-3.5 text-stone-400" />
              <span>
                Execution Context: Built-in synthetic public-safe profile{" "}
                <code className="font-mono text-stone-700 font-medium">
                  {selectedWorkflow.syntheticProfile}
                </code>
              </span>
            </div>
          </div>

          {/* Right Col: Structured Parameters & Turnstile Execution */}
          <div className="bg-white border border-[#E5E5E2] rounded-xl p-5 flex flex-col justify-between gap-5 text-left shadow-sm">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-2 border-b border-stone-100 pb-3">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span className="font-semibold text-sm text-stone-900">Parameters</span>
              </div>

              <div className="flex flex-col gap-3 text-xs">
                <div className="flex flex-col gap-1">
                  <label className="text-stone-600 font-medium">Materiality Tier</label>
                  <select
                    value={materiality}
                    onChange={(e: any) => setMateriality(e.target.value)}
                    className="border border-[#E5E5E2] rounded px-2.5 py-1.5 text-stone-800 bg-stone-50/50 outline-none focus:border-indigo-500 font-mono"
                  >
                    <option value="high">High Materiality (Tier 1 Model)</option>
                    <option value="medium">Medium Materiality (Tier 2)</option>
                    <option value="low">Low Materiality (Exploratory)</option>
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-stone-600 font-medium">Trial Budget</label>
                  <input
                    type="number"
                    value={trialCount}
                    onChange={(e) => setTrialCount(Number(e.target.value))}
                    min={5}
                    max={100}
                    className="border border-[#E5E5E2] rounded px-2.5 py-1.5 text-stone-800 bg-stone-50/50 outline-none focus:border-indigo-500 font-mono"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-stone-600 font-medium">Deterministic Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    className="border border-[#E5E5E2] rounded px-2.5 py-1.5 text-stone-800 bg-stone-50/50 outline-none focus:border-indigo-500 font-mono"
                  />
                </div>
              </div>
            </div>

            {/* Turnstile Verification Widget */}
            <div className="flex flex-col gap-3 pt-2 border-t border-stone-100">
              <TurnstileWidget onTokenReceived={(tok) => setTurnstileToken(tok)} />

              <button
                data-testid="run-start-workbench-button"
                onClick={handleLaunch}
                disabled={isRunning}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs tracking-wide shadow-sm disabled:opacity-50 transition-colors cursor-pointer"
              >
                <span>Run StART Workbench</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
