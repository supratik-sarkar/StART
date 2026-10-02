# StART Web Workbench — Quick Guide

The StART Web Workbench is the browser-based interface for exploring evidence-native model development and review.

Open the public Workbench:
https://start-mrt-gateway.sapman.workers.dev

No installation or API key is required for the core public workflow. Public rate limits may apply.

---

## 60-second workflow

Most first-time users only need this:

1. **Describe the objective**
   Enter what you want to evaluate in the objective box.

2. **Choose a workflow**
   Select the model/risk domain you want StART to execute.

3. **Choose an execution dataset**
   Pick one of the compatible public datasets.

4. **Click Build plan**
   StART prepares the execution plan before anything is run.

5. **Inspect the plan**
   Review the proposed stages under Inspect before execution.

6. **Click Accept & execute plan**
   StART runs the deterministic workflow and produces the review evidence.

7. **Explore the result**
   Use the Workbench, Data Runtime, and Scientific Certification surfaces to inspect what happened and why.

That's the core flow:

```text
Objective ↓ Workflow + Dataset ↓ Build plan ↓ Inspect ↓ Accept & execute ↓ Evidence-backed review
```

---

## Control map

### Main surfaces

| Control | What it is for |
| :--- | :--- |
| Workspace | Main review surface. Start a workflow, inspect the resulting run, and navigate its evidence and analysis. |
| Data Runtime | Inspect the execution/data-runtime perspective for the active workflow. |
| Scientific Certification | Inspect the scientific evidence and validation/certification surface associated with the run. |

### Workflow setup

#### Objective
The large text box answers:
*What are you engineering or evaluating?*
Be specific enough that the execution plan has a clear purpose.

**Good:**
Evaluate the predictive model on institutional credit data and show the evidence required before review.

**Less useful:**
Run ML.

#### Domain / executable workflow
This selects what kind of deterministic workflow StART should run.

Examples available in the public Workbench include domains such as:
- predictive ML;
- quantitative finance;
- supported scientific/model-review workflows exposed by the current capability catalog.

Changing the workflow also changes which execution datasets are compatible.

#### Canonical execution dataset
Select the dataset/context against which the workflow should execute.
Only compatible public execution contexts are shown.

The dataset selector is not merely cosmetic: it determines the actual data contract used by the workflow.

#### Build plan
Use this before execution.
Build plan does not mean "blindly run the model."
It asks StART to construct the review/execution plan first.

After clicking it, look for:
**Inspect before execution**
Review the proposed stages before continuing.

#### Accept & execute plan
Use this when the proposed plan matches what you intended to evaluate.
This is the point where the protected execution begins.

The public site performs its anti-abuse verification automatically in the background during normal use.
You do not need to provide an OpenAI, Anthropic, or other paid API key.

---

## Reading a completed run

After execution, StART switches from setup into investigation/review.

Depending on the workflow, the available analytical sections may include surfaces such as:
- Overview
- Universe & Data
- Construction
- Allocation
- Risk
- Performance
- Sensitivity
- Stability

Think of these as different views over the same run rather than independent analyses.

The key principle is:
> Agents may explain. Deterministic engines calculate. EvidenceRecords carry the quantitative truth.

Whenever interpretation and calculated evidence differ in purpose, StART keeps those responsibilities separate.

---

## After a run

### New workspace
Starts another review from a clean workspace.
Use this when you want to evaluate a different objective, dataset, or workflow.
A new protected execution obtains a new verification token automatically.

### Rerun
Use Rerun when you intentionally want to execute the current setup again.
This is useful for repeated evaluation of the same workflow context.
Always inspect the active objective and context before rerunning.

### Compare
Use Compare when you have multiple runs and want to inspect differences between them.
Typical uses include comparing:
- two executions;
- changed model assumptions;
- changed execution contexts;
- changed review outcomes.

If you only have one run, create another meaningful run before using comparison.

### Capability & evaluation catalog
Open the Capability & evaluation catalog when you want to see what the Workbench actually supports rather than guessing.
Use it to inspect the registered evaluation capabilities exposed by the running StART backend.

This is the safest answer to:
*"Can StART test this?"*

Do not assume that a control or algorithm exists simply because it would be useful.

---

## Example 1 — Predictive ML / credit-risk review

### Objective
Enter:
`Evaluate predictive_ml on institutional credit data and show the deterministic evidence required before accepting the model review.`

### Select
- **Workflow**: Predictive ML
- **Execution dataset**: `institutional_credit_v1`

### Then
1. Click **Build plan**.
2. Read **Inspect before execution**.
3. Click **Accept & execute plan**.
4. Wait for the run to complete.
5. Inspect the resulting evidence and investigation surfaces.

This is a good first run because it demonstrates the full:
```text
objective → plan → execution → evidence → review
```
cycle.

---

## Example 2 — Quantitative finance / portfolio-risk review

### Objective
Enter:
`Evaluate a risk-aware multi-asset portfolio. Keep portfolio weights separate from risk contributions and show the deterministic risk evidence.`

### Select
- **Workflow**: Quantitative Finance
- **Execution dataset**: `institutional_market_v1`

### Then
1. Click **Build plan**.
2. Inspect the proposed quantitative workflow.
3. Click **Accept & execute plan**.
4. Open the resulting analytical sections.

Useful sections to inspect include:
- Universe & Data
- Construction
- Allocation
- Risk
- Performance
- Sensitivity
- Stability

Pay particular attention to the distinction between:
**portfolio weight**
and
**risk contribution**

They are not the same quantity.

---

## Example 3 — Run the same review twice

1. Run a normal predictive-ML review.
2. After completion:
   Click **New workspace**.
3. Select the same workflow and execution dataset.
4. Enter a second objective or repeat the original objective intentionally.
5. Click **Build plan**.
6. Click **Accept & execute plan**.

The public Workbench treats the second protected execution independently.
Use this pattern before exploring Compare.

---

## Example 4 — Explore before executing

You do not need to execute immediately.
A useful exploration workflow is:
1. Open the Workbench.
2. Browse **Capability & evaluation catalog**.
3. Select a workflow.
4. Select a compatible dataset.
5. Enter an objective.
6. Click **Build plan**.
7. Stop at **Inspect before execution**.

This lets you understand what StART intends to do before allowing execution.

---

## What the status indicators mean

- **DEFINE**: You are specifying the problem, workflow, data context, or execution plan.
- **EXECUTE**: The accepted workflow is being run by the deterministic execution layer.
- **INVESTIGATE**: Execution evidence is available and you are examining the resulting review.

### AI provider shows Offline — is that an error?
No.
The public Workbench does not require a paid hosted LLM for its core public workflow.
You may see an AI/provider indicator showing an optional hosted provider as offline.
That does not mean the deterministic Workbench is unavailable.

StART deliberately keeps quantitative execution independent from hosted LLM availability.

### Turnstile / browser verification
The public portal uses Cloudflare Turnstile as an anti-abuse control.
During ordinary use it should operate in the background without an intrusive challenge.

You should not need to:
- provide an API key;
- create an account;
- configure Turnstile yourself.

Each protected execution obtains fresh verification automatically.

### If a button appears unavailable
Before assuming something is broken, check:
- Did you enter an objective?
- Did you select a workflow?
- Did you select a compatible execution dataset?
- If you already built a plan, are you looking for **Accept & execute plan** instead of **Build plan**?
- Are you currently inside a completed run rather than a new workspace?

For a clean restart, use:
**New workspace**

### If the public service is rate-limited
The public Workbench runs on shared public infrastructure.
If you receive an HTTP 429 or an explicit rate-limit message:
- do not repeatedly click the action;
- wait briefly;
- retry the same operation.

A rate limit is not a scientific/workflow failure.

---

## A useful mental model

When using StART, keep these layers separate:

```text
You define the objective ↓ StART constructs the plan ↓ You inspect / accept the plan ↓ Deterministic engines execute ↓ EvidenceRecords capture results ↓ Review surfaces explain the evidence ↓ Governance / policy remain separate
```

The browser is the workbench.
The evidence—not the prose—is the quantitative authority.

---

## Start here

Public Workbench:
https://start-mrt-gateway.sapman.workers.dev

For architecture, installation, CLI workflows, and deeper technical detail, return to the [main StART README](../README.md).
