# StART — Public Language Map & Terminology Specification

This document defines the linguistic and semantic transformation of **StART** from an internal model validation/review tool into the **Agentic AI Engineering Workbench** (*Build · Tune · Stress · Explain · Compare · Govern*).

---

## 1. Executive Terminology Rationale

StART was historically framed around model review, regulatory compliance, and independent model validation (SR 11-7 / OCC 2011-12). While governance and evidence sealing remain rigorous core competencies, modern AI engineering teams require a workbench for active development, tuning, stress-testing, and iterative experimentation.

The **Public Terminology Profile** re-anchors the primary user experience around building, tuning, challenging, and experimenting.
The **Enterprise / Model Risk Management (MRM) Profile** preserves the regulatory language for risk officers, compliance teams, and validation auditors without diluting the primary engineering developer experience.

---

## 2. Core Terminology Mapping Table

| Legacy Concept | Public Workbench Profile (Default) | Enterprise / MRM Profile (Optional) | Context & UI Location |
| :--- | :--- | :--- | :--- |
| **Review** (noun) | **Workspace / Session / Run** | Model Validation Review | WorkbenchHeader, HistoryRail, Session tabs |
| **New Review** | **New Workspace / New Session** | New Model Review | Primary top-level creation action button |
| **Validate** (verb) | **Build & Run / Execute Plan** | Validate Model / Challenge Model | Composer submit button, Action triggers |
| **Reviewer** (noun) | **Workbench Agent / Engineering Agent** | Model Validator / Independent Reviewer | Agent committee cards, Agent role badges |
| **Validation Run** | **Engineering Run / Experiment Run** | Validation Assessment Run | Execution state badges, logs, history |
| **Review In Progress** | **Executing Plan / Session Active** | Validation In Progress | Live execution status spinner/bar |
| **Validation Complete** | **Run Complete / Execution Finished** | Validation Completed | Terminal run state indicator |
| **Validation Plan** | **Build & Run Plan / Engineering Plan** | Model Validation Plan | Plan drawer, Plan Preview modal |
| **Evidence Locker** | **Artifact & Evidence Vault** | Evidence Locker / Audit Archive | Evidence tab, cryptographic artifact pane |
| **Evidence Record** | **Evidence Record** (preserved) | Evidence Record (preserved) | Deterministic hashed audit record |
| **Findings** | **Findings & Diagnostic Insights** | Model Validation Findings (MRA/MRIA) | Findings table, severity badges |
| **Validation History** | **Run History / Experiment History** | Validation History | Left sidebar rail, comparison selector |
| **Catalog** | **Capability & Evaluation Catalog** | Test & Challenge Catalog | Capability browser, test selection grid |
| **Challenger Run** | **Challenger Benchmark / Candidate Model** | Challenger Model Assessment | Model comparison view |
| **Sign-off / Approval** | **Sign-off & Seal / Acceptance Receipt** | Supervisory Sign-off / MRA Sign-off | Governance closure modal |

---

## 3. UI Component Mapping Details

### 3.1. Header & Navigation ()
- **Brand Subtitle**: `Agentic AI Engineering Workbench` (replacing `Enterprise Model Validation`).
- **Pill Badges**:
  - Execution Mode: `HYBRID WORKBENCH` (with switcher to `AGENTIC SESSION` / `DETERMINISTIC RUN`).
  - Provider Status: `AI · OpenAI · gpt-5.1 · Ready`.
  - Profile Toggle: `Workbench Profile` vs `Enterprise MRM Profile`.
- **Primary CTA**: `+ New Workspace` (replacing `+ New Review`).

### 3.2. Configuration & Composer ()
- **Objective Input**: `Engineering Objective & Hypothesis` (replacing `Review Scope & Instructions`).
- **Domain Selector**: Predictive ML, Deep Learning, Fraud / Anomaly, Recommenders, Quant & Risk, LLM & Agents.
- **Action Buttons**:
  - `Generate Plan (AI)`: Calls `gpt-5.1` to formulate an end-to-end build, stress, and test plan.
  - `Execute Plan`: Runs the deterministic science engines.
  - `Direct Run`: Deterministic pipeline run without AI synthesis.

### 3.3. Canvas & Evidence Presentation (, )
- **Overview Card**: Shows Model Architecture, Hyperparameter Profile, Cross-Validation Scores, and Stress Frontier.
- **Diagnostics**: Convergence curves, confusion matrices, SHAP summary plots, sensitivity response curves (-30% to +30%).
- **Vault Pane**: Artifact & Evidence Vault with SHA-256 integrity hashes and cryptographic signatures.

### 3.4. History & Replay ()
- **Run Cards**: Labeled with timestamp, dataset, model choice, primary score (AUC/NDCG/Sharpe), and status (`RUN COMPLETE`).
- **Compare Action**: Select two runs to open the side-by-side Engineering Run Comparator.

---

## 4. Implementation Guidelines for Dual-Profile Support

To support both personas seamlessly:
```typescript
export type TerminologyProfile = 'workbench' | 'enterprise';

export interface TerminologyMap {
  workspaceTitle: string;
  createAction: string;
  executeAction: string;
  planTitle: string;
  historyTitle: string;
  vaultTitle: string;
  agentRoleTitle: string;
}

export const TERMINOLOGY: Record<TerminologyProfile, TerminologyMap> = {
  workbench: {
    workspaceTitle: 'AI Engineering Workspace',
    createAction: 'New Workspace',
    executeAction: 'Build & Run',
    planTitle: 'Engineering Build Plan',
    historyTitle: 'Run History',
    vaultTitle: 'Artifact & Evidence Vault',
    agentRoleTitle: 'Workbench Agent'
  },
  enterprise: {
    workspaceTitle: 'Model Validation Workspace',
    createAction: 'New Review',
    executeAction: 'Validate Model',
    planTitle: 'Model Validation Plan',
    historyTitle: 'Validation History',
    vaultTitle: 'Evidence Locker',
    agentRoleTitle: 'Model Validator'
  }
};
```

The application defaults to `workbench` profile across all user-facing surfaces. When switched to `enterprise`, regulatory nomenclature is displayed without altering any backend API contracts or evidence records.
