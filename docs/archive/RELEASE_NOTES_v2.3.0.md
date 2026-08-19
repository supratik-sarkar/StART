# StART v2.3.0 — Evidence-Driven Review Committee

v2.3.0 transforms StART from a model-review pipeline into an evidence-driven
model-risk review committee: every recommendation is presented with the
evidence behind it, every reviewer challenge is tracked, diagnostic questions
are answered only from artifacts (never fabricated), and the governance
decision reads like a genuine MRM sign-off. This is an enhancement release —
all v2.2.0 functionality is preserved.

## Highlights

### Evidence-constrained agent dialogue (anti-hallucination)
Ask-Agent now follows Question -> Artifact Retrieval -> Evidence Assembly ->
Answer -> Critic. Diagnostic questions (outliers, missingness, correlations,
feature importance, sensitivity, cohort metrics, tuning, leakage) are answered
ONLY from the evidence store and never reach an LLM. When the evidence exists,
real values are returned with provenance; when it does not, the agent replies
"I do not have sufficient evidence to answer this question." No feature names,
percentages, thresholds, counts, or drift values are ever fabricated — even a
connected LLM that tries to invent diagnostics is bypassed.

### Reviewer challenge memory
Reviewer challenges ("Why not WideDeep?", "I disagree with correlation
pruning", "Show sensitivity evidence") persist through the review with
timestamp, originating agent, evidence used, response, and status
(open/closed/unresolved). They surface in the terminal, dashboard, transcript,
and notebook, and feed the sign-off decision.

### Committee cards
Every agent interaction renders as a committee review card with the mandatory
evidence-first structure: Evidence -> Recommendation -> Alternatives -> Risks ->
Artifacts -> Open questions -> Decision. If a recommendation lacks supporting
evidence, the card states "The recommendation is not evidence-backed."

### Dataset discovery & feature-engineering transparency
Before any recommendation, DatasetDiscoveryAgent shows detected/candidate
targets, feature inventory, class balance, missingness, outliers, leakage
candidates, correlation summary, and the proposed train/test/OOS split.
FeatureEngineeringAgent shows the real diagnostics behind each decision
(outlier counts/percent/rule/action, correlation pairs/coefficients/proposed
drop) — real values, not placeholders.

### ValidationAgent + full sensitivity review
ValidationAgent is a first-class checkpoint before sign-off. It presents a
feature sensitivity ranking, a shock table (-30%..+30%), business
interpretation of the most sensitive features, and the sign-off impact, with
[A] accept / [Q] ask / [C] challenge. Sensitivity questions are answered from
the sensitivity artifacts.

### MRM-grade sign-off
GovernanceSignoffAgent weighs performance, generalization, calibration,
feature dependence (sensitivity), reviewer challenges, and overrides, and
returns READY / READY WITH CONDITIONS / NOT READY with an evidence-cited
factor table. Excessive feature dependence cannot silently receive READY.

### Terminal Rich-table standardization
The terminal uses Rich tables exclusively (metrics, tuning, importance,
sensitivity, FE diagnostics, adapter inventory, challenge log, sign-off).
Markdown tables remain only in dashboard.md, reports, and transcripts.

### Adapter transparency
Each adapter shows purpose, status, execution time, artifacts produced, and
evidence produced in a Rich inventory table.

### Review journey surfacing
Challenges, MRM sign-off, ValidationAgent review, and the committee transcript
are embedded in the dashboard and transcript, and mirrored in notebook 05 —
same engine, no duplicate logic.

## Enterprise LLM gateway UX

A first-class gateway-first prompt now separates enterprise mode from public
paid-key mode.

- Interactive terminal runs (propensity and DL, via the shared
  `prompt_review_config`) ask "Use enterprise LLM gateway? [y/N]" before any
  provider choice. Answering yes sets provider `enterprise_llm_gateway` and the
  enterprise trust domain, skips the public provider menu, and never requests
  OpenAI/Anthropic/Grok keys. Answering no preserves the existing public flow
  (none/openai/anthropic/grok with the hidden key prompt).
- Notebook 05 mirrors this via an `enterprise_gateway` widget that, when set to
  yes, routes through the gateway and skips public keys.
- If the private enterprise package/env is absent, the gateway degrades
  explicitly to deterministic review — it never silently falls back to a public
  provider.
- Env-var alias: `START_ENTERPRISE_LLM_PACKAGE` is now accepted (preferred),
  with backward compatibility for `START_ENTERPRISE_PACKAGE`.

### Trust-domain isolation (unchanged, re-verified)
Public providers and the enterprise gateway remain strictly isolated: no
crossover (enforced at the routing boundary in both directions), no key
sharing, and no cross-domain fallback. The enterprise gateway is a generic,
abstract route — the public repository contains no firm name, endpoint,
credential, or hostname, and the real implementation is loaded from a private,
firm-approved package inside the firm environment with no structural change to
the public code. The design uses only `os.environ`, `importlib`, and `pathlib`,
so it works on Windows PowerShell/Command Prompt, Linux shells, and
Databricks-like runtimes.

## Compatibility
All v2.2.0 behavior is preserved; v2.3.0 is additive.
