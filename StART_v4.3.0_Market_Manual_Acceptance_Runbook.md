# StART v4.3.0 — Market Manual Interactive Acceptance Runbook
## Post–Gate 13B Live GPT-5 Re-Test

Use this file as a copy-paste checklist during the manual terminal test.

---

# 0. Fresh Terminal Setup

```bash
cd ~/StART
source .venv-start/bin/activate
export PS1="StART % "
clear
start review
```

---

# 1. Review Setup Selections

## Select Review Mode
Paste:

```text
1
```

Expected: **Single-Domain Review**

## Select Review Domain
Paste:

```text
2
```

Expected: **Market Risk & Portfolio Analytics**

## Select AI Reviewer Agent Backend
Paste:

```text
3
```

Expected: **Public LLM Providers**

## Select Public LLM Provider
Paste:

```text
1
```

Expected: **OpenAI**

## Select OpenAI Model
Paste:

```text
2
```

Expected: **gpt-5**

## Select Model Materiality
Paste:

```text
1
```

Expected: **High (Tier 1)**

## Select Review Lifecycle
Paste:

```text
1
```

Expected: **Initial Validation**

---

# 2. Governance Information

## Business Context

```text
This is an independent validation of a high-materiality market risk and portfolio analytics framework used to assess the daily risk profile of a diversified multi-asset portfolio.

The framework supports portfolio construction challenge, volatility and diversification analysis, covariance estimation, factor attribution, VaR and tail-risk assessment, scenario and reverse-stress analysis, risk monitoring, escalation, and governance reporting.

The review should assess methodological soundness, numerical stability, data integrity, portfolio constraints, covariance structure, attribution reconciliation, VaR exception behavior, tail dependence, scenario sensitivity, and traceability.

Successful execution or failure to reject an individual statistical test is not sufficient for model validation. Unsupported assumptions, unstable estimates, unresolved sensitivities, unexplained residuals, negative validation evidence, or insufficient governance criteria must remain visible.

Deterministic StART engines are the source of quantitative results. The AI reviewer may interpret and challenge those results but must not invent arithmetic, thresholds, or unsupported conclusions.

END
```

## Reviewer Clarification

```text
Act as an independent model-risk reviewer.

Prioritize methodological validity, implementation correctness, numerical stability, sensitivity, reproducibility, interpretability, and auditability.

Challenge return definitions, annualization assumptions, portfolio constraints, covariance conditioning, factor specification and reconciliation, VaR sign conventions, exception dependence, missing-data treatment, optimization stability, scenario assumptions, and reverse-stress interpretation.

Failure to reject a statistical test is not equivalent to model correctness.

Solver convergence is not equivalent to scientific validity.

PSD repair is not equivalent to covariance-model validation.

RECORDED is not equivalent to PASS.

Distinguish deterministic evidence, methodological interpretation, unresolved model risk, and conclusions that would require an explicit external materiality criterion.

Do not invent numerical thresholds or external standards.

Every quantitative claim must be grounded in the EvidenceRecords supplied to the checkpoint.

Preserve negative, skipped, unresolved, and conditional evidence.

END
```

## Intended Use / Decision Impact

```text
Daily independent portfolio-risk assessment, allocation challenge, limit monitoring, VaR and tail-risk review, factor-risk assessment, scenario analysis, escalation, remediation prioritization, and governance.

Outputs may influence continued model use, portfolio-risk decisions, model remediation, and management escalation.

Consequential conclusions must distinguish deterministic measurements, statistical validation results, reviewer interpretation, unresolved uncertainty, and formal governance decisions.

END
```

## Known Limitations / Reviewer Concerns

```text
Potential limitations include covariance instability, optimizer sensitivity, factor misspecification, attribution residuals, VaR exception clustering, missing-data sensitivity, scenario dependence, reverse-stress interpretation, and numerical degeneracy.

Shrinkage, regularization, or PSD repair should not automatically be interpreted as model validation.

VaR non-rejection alone is insufficient for approval.

Scenario severity and reverse-stress distance should not be labelled material, acceptable, or unacceptable unless an explicit criterion exists.

Challenge skipped diagnostics, unsupported assumptions, weak evidence, unresolved contradictions, and insufficient provenance.

AI reasoning may be qualitative, but every quantitative statement must trace to a supplied EvidenceRecord and metric.

END
```

---

# 3. Market Data & Scope

## Select Market/Treasury Data Source

```text
1
```

Expected: **Built-in Synthetic Market World**

## Select Review Scope

```text
1
```

Expected: **Full Recommended Review**

## Proceed to Execute Review

```text
y
```

---

# 4. Portfolio Risk & Volatility Assumptions

This checkpoint was already manually proven to support V/Q/C/A correctly in the previous run.

Paste:

```text
a
```

Expected: advance to **Factor Modeling & Attribution Assumptions**

---

# 5. Factor Modeling & Attribution Assumptions

## View checkpoint artifacts

```text
v
```

Expected: a Factor / Attribution artifact backed by existing attribution EvidenceRecords, without new analytical recomputation.

## Ask reviewer

```text
q
```

Then paste:

```text
Based only on the attribution evidence at this checkpoint, assess whether return and risk attribution reconcile adequately. Identify the strongest residual concern involving factor specification, unexplained contribution, or method dependence, and distinguish observed evidence from assumptions that remain unvalidated.
```

Expected:
- Real GPT-5 prose.
- Relevant structured attribution evidence.
- Reviewer should not be restricted only to `attribution.cross_sectional_factor_model` if other legitimate attribution records are present.
- Claim grounding passes if quantitative claims are made.

If successful:

```text
a
```

---

# 6. VaR Backtesting & Exception Frequency — CRITICAL

Before typing anything, inspect the Rich table.

Expected scientific content approximately:

```text
VaR exceptions: 6 / 1000
Exception rate: about 0.006
VaR confidence: 0.99
Kupiec LR: about 1.886 (NOT 1.850)
Kupiec p-value: about 0.1696
Kupiec decision: DO_NOT_REJECT
Hypothesis-test significance: gamma=0.05
validation.var_size_power size: 0.066
Pre-registered band: [0.031, 0.069]
Power values: about 1.000 and 0.992
Validation status: PASS
```

Critical invariant:

```text
confidence = 0.99
alpha_var = 0.01
gamma_test = 0.05
```

The terminal must NOT show `gamma=0.99` or `gamma=0.01` for the hypothesis tests.

## View VaR artifact

```text
v
```

## Ask the critical VaR question

```text
q
```

Then paste:

```text
Interpret the VaR backtesting evidence without treating failure to reject as proof that the VaR model is correct. Explain what unconditional coverage, exception dependence, conditional coverage, tail severity, and any skipped diagnostic jointly imply, using only supplied evidence.
```

Expected:
- Real GPT-5 response.
- Correct EV citations.
- No `1.850` fallback statistic.
- No conflation of confidence / alpha_var / gamma_test.
- Claim Grounding Gate: PASSED.

Grounding census invariant:

```text
Grounded + Unbound = Quantitative Claims
```

Example valid result:

```text
Quantitative claims: 12 | Grounded: 12 | Unbound: 0
```

## Challenge VaR result
Only if Q passes.

```text
c
```

Then paste:

```text
Challenge the VaR result using the registered non-mutating exception-duration or dependence diagnostic. Determine whether clustering or inter-exception timing reveals model risk that unconditional exception frequency alone would miss. Preserve the distinction between statistical evidence and model acceptance.
```

Expected:
- Deterministic non-mutating diagnostic.
- New diagnostic EvidenceRecord.
- Grounded reviewer interpretation.
- Remain on same checkpoint.

If successful:

```text
a
```

---

# 7. Covariance Structure & Missing Data Treatment

## View artifacts

```text
v
```

## Ask reviewer

```text
q
```

Then paste:

```text
Compare the empirical, shrinkage, and missing-data covariance evidence available at this checkpoint. Which numerical or structural property deserves the greatest scrutiny before relying on the covariance matrix for portfolio-risk or optimization conclusions? Do not assume that shrinkage, regularization, or positive-semidefinite repair establishes model validity.
```

## Challenge covariance

```text
c
```

Then paste:

```text
Run the applicable non-mutating covariance diagnostic and assess matrix conditioning, eigenvalue structure, numerical rank, and related stability evidence. Do not repair or alter the covariance matrix as part of this challenge.
```

Expected:
- Diagnostic only.
- No hidden Higham repair.
- No covariance mutation.
- Same checkpoint afterward.

If successful:

```text
a
```

---

# 8. Scenario Analysis & Stress Testing

Expected actual scenario evidence:

```text
scenario.linear_return
scenario.factor_linear
scenario.reverse_stress
```

Legacy read alias may exist:

```text
scenario.asset_return
```

Canonical new-write identity should remain:

```text
scenario.linear_return
```

## View scenario artifacts

```text
v
```

## Ask reviewer

```text
q
```

Then paste:

```text
Using only the scenario, stress, and reverse-stress evidence generated in this run, explain what these results reveal beyond the VaR evidence. Separate deterministic scenario loss or sensitivity measurements from any judgement that would require an external materiality threshold.
```

## Challenge scenario

```text
c
```

Then paste:

```text
Challenge the scenario conclusions using the registered deterministic scenario-data-integrity diagnostic. Verify shock completeness, units, factor or asset alignment, and admissibility without changing the scenario or silently repairing any input.
```

Expected:
- Non-mutating diagnostic.
- No scenario repair/recomputation.
- Same checkpoint afterward.

If successful:

```text
a
```

---

# 9. Cross-Analytical Committee Synthesis

## View all artifacts

```text
va
```

## Ask committee

```text
q
```

Then paste:

```text
Synthesize the strongest cross-analytical model-risk concern across portfolio construction, covariance, attribution, VaR, and scenario evidence. Identify where individually reasonable results do not justify unconditional approval when considered jointly. Separate deterministic evidence, methodological dependency, unresolved risk, and conclusions that require an external materiality criterion.
```

## Challenge committee synthesis

```text
c
```

Then paste:

```text
Challenge the committee synthesis. Determine whether the strongest unresolved cross-analytical concern can actually be resolved using a registered deterministic diagnostic or whether it must remain evidence-only because no applicable materiality criterion exists. Do not manufacture a contradiction merely because two methods differ.
```

Expected:
- Legitimate deterministic resolution OR evidence-only resolution.
- No invented threshold.
- No false contradiction from method disagreement.

If successful:

```text
a
```

---

# 10. Barrier Validation

For the standard synthetic Market run, this checkpoint should ideally be omitted if barrier evidence is only SKIPPED / N/A.

If it DOES NOT appear: good.

If it appears with only SKIPPED/N/A evidence: STOP and capture output.

If it appears with real applicable evidence:

```text
v
```

Then:

```text
q
```

Then paste:

```text
Explain exactly what the Brownian-bridge barrier evidence establishes and what it does not establish. Distinguish deterministic boundary-crossing or admissibility evidence from broader conclusions about portfolio or model validity.
```

If successful:

```text
a
```

---

# 11. Model Governance & Attestation Sign-Off

## View all artifacts

```text
va
```

## Ask governance reviewer

```text
q
```

Then paste:

```text
Before final sign-off, summarize the strongest evidence supporting continued use and the strongest unresolved evidence arguing against unconditional approval. Separate implementation verification, deterministic analytical findings, statistical validation results, unresolved model risk, and governance criteria.
```

## Challenge unconditional approval

```text
c
```

Then paste:

```text
Challenge unconditional approval of this Market Risk review. Preserve all negative, conditional, skipped, and unresolved evidence. Do not treat successful execution, solver convergence, absence of a statistical rejection, or artifact generation as sufficient evidence of model validity.
```

Expected:
- Negative evidence preserved.
- Same checkpoint afterward.
- `ACCEPT_WITH_CONDITIONS` is acceptable if scientifically warranted.
- No forced unconditional PASS.

## Final Accept
Only if the full run is coherent:

```text
a
```

---

# 12. Criterion Provenance Watch

During Committee / Governance, watch for:

```text
PRE_REGISTERED_VALIDATION
```

versus:

```text
STATISTICAL_TEST_SPECIFICATION
```

A root VaR test may legitimately carry a local engine-level criterion source such as `STATISTICAL_TEST_SPECIFICATION`.

But Governance must not incorrectly replace the established statistical-validation/governance provenance `PRE_REGISTERED_VALIDATION` with a lower-level engine-specification label.

If Committee or Governance semantically substitutes one for the other, capture the output before accepting.

---

# 13. Do NOT Use Override in Main Acceptance Run

Do NOT use:

```text
o
```

in the principal acceptance run.

If override UX is tested later in a disposable run, suggested rationale:

```text
Reviewer override entered solely to verify interactive override provenance and state transition behavior. This is a workflow-validation action and must not be interpreted as a substantive change to the underlying deterministic model-risk evidence.
```

---

# 14. Immediate Stop Conditions

STOP the run and capture terminal output if ANY of these occurs:

1. GPT-5 returns empty / whitespace.
2. Provider response is incomplete, error, or refusal.
3. Grounding fails on a correctly cited response.
4. Grounded > Quantitative Claims.
5. Kupiec table shows LR=1.850.
6. validation.var_size_power shows size=N/A despite evidence existing.
7. gamma_test is shown as 0.99 or 0.01 for the 99% VaR test.
8. Q advances the checkpoint.
9. C advances the checkpoint.
10. V or VA advances the checkpoint.
11. C mutates or repairs covariance/scenario science.
12. Scenario checkpoint is empty.
13. Barrier checkpoint appears with only SKIPPED/N/A evidence.
14. Committee omits Scenario evidence.
15. Governance loses negative/unresolved evidence.
16. Committee/Governance invent materiality thresholds.
17. UI recomputes deterministic PASS/FAIL.
18. Artifact generation triggers a new analytical engine computation.

When a live reviewer / grounding failure menu appears, choose:

```text
2
```

to abort.

Do NOT continue deterministically during manual live-provider acceptance.

---

# 15. Action State-Machine Invariant

These must remain on the SAME checkpoint:

```text
Q
C
V
VA
```

Only these should advance:

```text
A
O
```

Use `A` in the main acceptance run; avoid `O`.

---

# 16. Frozen Scientific Invariants

- Agents orchestrate.
- Deterministic engines compute.
- Evidence is the product.
- No LLM arithmetic.
- RECORDED != PASS.
- Solver convergence != scientific validity.
- Failure to reject != model correctness.
- PSD repair != covariance-model validation.
- No hidden numerical repair.
- No invented materiality thresholds.
- No silent scenario repair.
- No silent provider/model fallback.
- `alpha_var != gamma_test != confidence`.
- Scenario new-write identity: `scenario.linear_return`.
- Legacy read alias: `scenario.asset_return`.
- Negative evidence remains visible.
- No criterion generally implies evidence-only resolution and conditional governance, not unconditional acceptance.

---

# 17. Expected Registry Counts

```text
Registered: 79
Unique: 79
Duplicates: 0
Predictive: 52
Market: 25
Treasury: 2
Market + Treasury: 27
```

---

# 18. Manual Acceptance Objective

The Market run is successful only if it proves:

1. real GPT-5 responses;
2. non-vacuous grounding;
3. canonical table/reviewer/grounding evidence consistency;
4. correct VaR semantics;
5. non-terminal Q/C/V/VA behavior;
6. deterministic challenges;
7. populated Scenario evidence;
8. correct artifact lineage;
9. Committee consumes all relevant checkpoint evidence;
10. Governance preserves negative evidence and uses scientifically valid conditional disposition where appropriate;
11. final attested narrative corresponds to the grounded narrative actually displayed.

Do not publish or synchronize Git merely because this Market run succeeds. Treasury / Cross-Domain / Predictive-DL acceptance and final authorized regression remain separate release steps.
