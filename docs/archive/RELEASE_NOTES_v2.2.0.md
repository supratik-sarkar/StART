# StART v2.2.0 — Interactive Review Committee

StART v2.2.0 turns the model-review pipeline into a live, collaborative review
environment. Instead of launching an automated pipeline, the user sits inside a
model-risk committee: a team of specialized agents that announce what they are
doing and why, answer questions, and honor the user's decisions throughout the
run. All v2.1.x engines are unchanged; this release makes the review visible,
interactive, and decision-driven.

## Highlights

### ReviewSession — one workflow engine
A single `ReviewSession` is the shared, persistent state of a review. It records
every checkpoint decision, agent conversation, override, and clarification in
order, and exposes queries (`rejected(...)`, `effective(...)`, `overrides()`,
`context_banner()`) that downstream agents and the dashboard read from. Both the
terminal and notebook drive the same engine, so the review is one experience
with two front ends — not two divergent workflows.

### Ask Agent — talk to the committee
At each checkpoint the user can press `[Q] Ask agent` and ask a free-form
question — "Why MLP?", "Show alternatives", "Why not XGBoost?", "What if I keep
all features?". When a provider is genuinely connected the agent answers in
natural language, grounded only in data-free decision context (its
recommendation, reasons, alternatives, and the dataset shape — never raw rows).
Otherwise it answers deterministically from the structured reasoning the agents
already produce. Every exchange is recorded in the review transcript. Ask Agent
is wired across target discovery, feature engineering, model recommendation,
hyperparameter tuning, sensitivity, and governance/signoff touchpoints.

### Committee roster — agents are first-class participants
At startup the run introduces the committee — DatasetDiscoveryAgent,
FeatureEngineeringAgent, ArchitectureReviewAgent, HyperparameterTuningAgent,
ModelExecutionAgent, ValidationAgent, GovernanceSignoffAgent, and
EvidenceCriticAgent — each with its purpose. Agents announce what they are
doing, why, what they recommend, and a confidence level.

### Adapter activity announcements — the ecosystem working in real time
Beyond the startup/end AI-engineering panel, adapters now announce their
activity live during execution, e.g. `[OPA] Validating policy controls…`,
`[DeepEval] Running quality checks…`, `[LangSmith] Capturing review trace…`,
`[Phoenix] Recording observability artifact…`. The panel itself shows each
adapter's purpose, status, and a check/cross for availability.

### Decision-driven execution — choices change the run, not just the record
User decisions actually affect what executes:
- Rejecting correlation pruning keeps all features; the model trains on the full
  feature set and the trace records that the rejection was honored. Not
  rejecting it drops highly-correlated features before training.
- Keeping a non-recommended architecture (e.g. wide_deep over MLP) makes the
  execution trace explicitly state the user override was honored — the model
  under review is the user's choice.
- Setting cost priority to false negatives routes tuning and validation to
  PR-AUC / recall.

### Sensitivity tables — visible everywhere
Feature-shock sensitivity is rendered as a table in the terminal, the notebook,
the dashboard, and the transcript, with columns: feature, shock %, baseline
metric, shocked metric, delta, and a governance risk-impact label. The 0% row
equals the baseline by construction.

### Dashboard review journey
The primary dashboard now embeds the committee transcript as a Review Journey
section — decisions, user overrides, and agent conversations — in addition to
the standalone `transcript.md` / `transcript.html` / `transcript.json` sibling
files. The dashboard reads like a committee review record, not just outputs.

### Notebook parity
Notebook 05 shares the same agents, control surface, sensitivity table, Ask-Agent
capability (programmatic where live stdin is not technically possible), and
writes the same transcript artifacts as the terminal.

## AI-engineering adapters

### LangSmith and Phoenix
LangSmith and Phoenix are integrated as optional enterprise observability and
evaluation adapters (Moonshot-safe alternatives). They are detected dynamically:
present in the environment, they appear as available and announce activity;
absent, they appear with install guidance. Detection is environment-driven, so
the same code reflects whatever is installed on the running machine.

### Moonshot exclusion rationale
Moonshot remains intentionally excluded from the default environment.
`aiverify-moonshot==0.7.6` hard-pins `pydantic==2.8.2` and
`huggingface-hub~=0.36`, which conflict with MCP, DeepEval, Garak, LiteLLM, and
Transformers. The adapter's control-surface metadata documents this and
recommends LangSmith or Phoenix instead. Moonshot is not required and should not
be reinstalled into the primary StART environment.

## Trust-domain isolation
Public providers (OpenAI, Anthropic, Grok) and the generic
`enterprise_llm_gateway` remain strictly isolated, with separate provider paths
and credential handling and no shared secret pathways. Keys are read from the
environment via a hidden prompt, session-only, never echoed or written to
artifacts. The enterprise gateway is intentionally generic — StART is a public
clean-room framework with zero firm-specific names, endpoints, or credentials.

## Compatibility
All v2.1.x behavior is preserved; v2.2.0 is additive. The full suite passes,
lint is clean across source, tests, examples, and notebooks, and the enterprise
review runs end to end.
