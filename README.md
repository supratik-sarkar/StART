# StART — Standardized Agentic Reusable Tests

StART is an **evidence-native model dev/review, risk management, and governance platform** designed for institutional machine learning, deep learning, and quantitative finance.

Unlike conventional LLM-based assistants that perform hallucination-prone arithmetic, StART enforces a strict architectural invariant: **AI agents reason and orchestrate, while deterministic mathematical engines perform all computations**. Every diagnostic produces an immutable, cryptographically signed `EvidenceRecord`, which is appended to a replayable hash-chained ledger and sealed into a Merkle tree attestation.

---

## Quick Start

### 1. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/supratik-sarkar/StART.git
cd StART

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv-start
source .venv-start/bin/activate

# Install package with all modeling and telemetry stacks
pip install -e ".[all]"

# Verify environment integrity
start doctor
```

### 2. Run Deterministic Workflows

```bash
# Run Predictive & Deep Learning development/review workflows
start review --domain predictive --mode deterministic

# Run Market & Portfolio analytical workflows
start review --domain market --mode deterministic

# Launch the local StART Web Workstation
python -m uvicorn start.web.app:app --port 8000

# Open http://localhost:8000 in a modern browser
```

---

## StART Institutional Workstation & Browser AI

The **StART Institutional Workstation** delivers an evidence-native interface designed for ML/AI engineering, software/technology teams in financial institutions, model development, independent review, quantitative analysis, and risk/governance.

* **Agentic Engineering Workspace Layout**:
  - **Task-Oriented Composer**: Initiate workflows across Predictive ML, Deep Learning, Calibration, Robustness, Explainability, Hyperparameter Tuning, Model Comparison, and Quantitative Finance.
  - **Live Execution & Findings**: Real-time structured progress tracking, runtime execution graphs, and evidence-grounded findings with contextual iterative actions (*Explain with AI*, *Challenge*, *Run deeper test*, *Compare candidates*, *Re-run*).
  - **Interactive Evidence & Artifact Inspector**: Dynamic React Flow evidence decision graphs, interactive charts (ROC curves, calibration distributions, SHAP attributions, efficient frontiers), deterministic PDF reports, and provenance JSON.

* **Client-Side WebLLM Reviewer (WebGPU Client Inference)**:
  - Executes local small language models (pinned model: `SmolLM2-1.7B-Instruct-q4f16_1-MLC`) **directly inside the user's browser via WebGPU** without blocking deterministic engine execution.
  - **Server-Side Hydration Protocol**: The browser LLM only cites Evidence IDs (`[EV-xxxx]`); the backend server rejects any client-supplied numbers, hydrates exact numerical metrics directly from immutable `EvidenceRecord`s, evaluates authentic **OPA** Rego policies, and generates the final Merkle attestation root.

* **Public Application**:
  - **Primary Public Workstation**: [https://start-mrt-gateway.sapman.workers.dev](https://start-mrt-gateway.sapman.workers.dev)

---

## Core Architecture Schematics

### 1. End-to-End Review Orchestration Flow

```mermaid
flowchart TD
    User["Portfolio / Model Specification"] --> Context["Review Context Bundle"]
    Context --> StateGraph["LangGraph StateGraph Engine"]
    StateGraph --> Specialist["Domain Specialist Agent"]
    Specialist --> Engines["79 Deterministic Analytical Engines"]
    Engines --> Ledger[("Cryptographic Evidence Ledger")]
    Engines --> Artifacts["Vector SVG & Tabular Artifacts"]
    Ledger --> StructRev["Structured Reviewer Graph"]
    StructRev --> Critic["Evidence Critic & Grounding Gate"]
    Critic --> Committee["Cross-Analytical Committee"]
    Committee --> OPA["OPA Policy & Security Plane"]
    OPA --> Seal["Merkle Root Attestation Seal"]
    Seal --> UI["Terminal / Presentation Model / Dashboards"]
```

### 2. LangGraph StateGraph & Checkpoint Persistence Flow

```mermaid
stateDiagram-v2
    [*] --> START
    START --> PlanNode: Initialize TypedReviewState
    PlanNode --> ExecuteToolsNode: Discover & Dispatch 79 Tools
    ExecuteToolsNode --> ReviewEvidenceNode: Commit EvidenceRecords
    ExecuteToolsNode --> ErrorRecovery: Exception / Validation Gap
    ErrorRecovery --> ExecuteToolsNode: Resume from Checkpoint (thread_id)
    ReviewEvidenceNode --> GenerateArtifactsNode: Render SVG / Tables
    GenerateArtifactsNode --> GovernanceSignoffNode: Committee Disposition
    GovernanceSignoffNode --> END: Merkle Root Attestation Seal
    END --> [*]
```

### 3. Open Policy Agent (OPA) Decision Boundary

```mermaid
flowchart LR
    Action["Runtime Request\n(Tool / Egress / Export / Signoff)"] --> PolicyPlane["OPA Policy Plane\n(opa eval / In-Process Engine)"]
    PolicyPlane --> RegoEgress["network_egress.rego"]
    PolicyPlane --> RegoTools["tool_allowlist.rego"]
    PolicyPlane --> RegoExport["artifact_export.rego"]
    PolicyPlane --> RegoGov["attestation.rego"]

    RegoEgress --> Decision{"Policy Decision"}
    RegoTools --> Decision
    RegoExport --> Decision
    RegoGov --> Decision

    Decision -- ALLOW --> Execute["Proceed with Execution"]
    Decision -- DENY --> Block["Fail-Closed Security Block"]
```

### 4. OpenTelemetry Hierarchical Span Trace

```mermaid
flowchart TD
    Run["review.run (Trace Root)"] --> Ckpt["review.checkpoint (Checkpoint Phase)"]
    Ckpt --> Agent["agent.execution (Specialist Agent)"]
    Agent --> Tool["tool.execution (Deterministic Tool)"]
    Tool --> Ev["evidence.commit (Evidence Ledger Append)"]
    Ev --> Art["artifact.generate (SVG/Table Render)"]
    Art --> Pol["policy.evaluate (OPA Decision)"]
    Pol --> Gov["governance.evaluate (Committee Review)"]
    Gov --> Seal["attestation.seal (Merkle Seal Signature)"]
```

### 5. Evidence & Attestation Lineage

```mermaid
flowchart LR
    DetResult["Deterministic Analytical Result"] --> EvRec["EvidenceRecord (SHA-256)"]
    EvRec --> HashChain["Append-Only Ledger Block"]
    HashChain --> CitRef["Reviewer Finding EvidenceMetricRef"]
    CitRef --> GraphHash["Finding Graph Merkle Leaf"]
    GraphHash --> MerkleRoot["Cryptographic Attestation Seal"]
```

---

## Key Technical Differentiators

* **79 Deterministic Validation Surfaces**: Comprehensive coverage across Portfolio Construction (MVO, HRP, HERC, MDP, Black-Litterman, CVaR LP), Factor Modeling, Covariance Conditioning, VaR Exception Backtesting (Kupiec, Christoffersen), Scenario Stress Repricing, Short-Rate Calibration (Vasicek, CIR, Hull-White), and PyTorch Tabular Deep Learning.
* **Provider-Neutral Structured Reviewer Contract**: LLMs reason over citations (`[EV-xxxx]`), but are mathematically barred from performing numerical arithmetic or inventing values.
* **100% Claim Grounding Verification**: Every numerical claim in the final review narrative is automatically validated against cited `EvidenceRecord` metrics before governance sign-off.
* **Resumable LangGraph StateGraph Runtime**: Production-grade compiled `StateGraph[TypedReviewState]` with typed state, conditional routing, checkpointer persistence (`MemorySaver`), failure recovery, and zero duplicate evidence on resume.
* **Open Policy Agent (OPA) Control Plane**: Strict fail-closed policy enforcement via authentic `.rego` policies for network egress, tool allowlists, agent permissions, artifact filtering, and attestation rules.
* **OpenTelemetry Observability**: Hierarchical spans (`review.run` $\to$ `checkpoint` $\to$ `agent` $\to$ `tool` $\to$ `evidence` $\to$ `governance` $\to$ `attestation`) with automated secret and credential redaction.
* **100% Offline & Private-Safe**: All core computations, graph executions, policy evaluations, and attestation seals execute locally in-process with zero network dependencies.

---

## Verified Architecture Capability Registry

| Component | Type | Classification | Verified Runtime Capability |
| :--- | :--- | :---: | :--- |
| **Deep Learning Institutional UX** | Deep Learning | `PROVEN_ADVANCED` | PyTorch tabular DL inspection, layer summaries, loss history, Optuna tuning, ECE calibration, SHAP, and SVG artifacts. |
| **StateGraph / LangGraph Runtime** | Orchestration | `PROVEN_ADVANCED` | Compiled StateGraph with typed state, conditional edges, checkpointers, resumability, and bounded retry. |
| **OpenTelemetry Tracing** | Telemetry | `PROVEN_ADVANCED` | Hierarchical span model with automated credential/secret redaction and in-memory export. |
| **Open Policy Agent (OPA)** | Policy | `PROVEN_ADVANCED` | Authentic OPA Rego evaluation with fail-closed policies for egress, tools, export filtering, and governance sign-off. |
| **NeMo Guardrails** | Security | `OPTIONAL_ADVANCED` | Real `RailsConfig`/`LLMRails` safety boundary, prompt injection defense, and EvidenceRecord immutability enforcement. |
| **LangSmith Tracer** | Telemetry | `OPTIONAL_ADVANCED` | Optional external telemetry exporter over canonical event model with strict redaction. |
| **MCP Server Integration** | Adapter | `OPTIONAL_ADVANCED` | Standardized Model Context Protocol adapter for tool discovery and typed capability inspection. |
| **Garak Vulnerability Scanner** | Security | `OPTIONAL_FUNCTIONAL` | Automated LLM vulnerability probing and adversarial prompt evaluation harness. |
| **Promptfoo / DeepEval** | Adapter | `OPTIONAL_FUNCTIONAL` | Unit-testing and regression evaluation harnesses for LLM prompt variations. |
| **Langfuse / Phoenix** | Telemetry | `OPTIONAL_FUNCTIONAL` | Trace capture and observability adapters consuming the unified event model. |

---

## License

Apache-2.0. Copyright (c) 2026 StART contributors.
