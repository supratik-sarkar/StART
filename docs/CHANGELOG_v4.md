# v4.0.0 — packaging revamp and the stripe-agnostic core

Major version because the packaging model and the Python floor both changed in
breaking ways. If you want the version string to read `3.3.0` instead, it is one
line in `pyproject.toml` and one in `src/start/__init__.py` — but semver says
this is a major.

## Breaking

- **Python 3.12+ required.** `requires-python`, ruff `target-version` and mypy
  `python_version` now agree; a contract test fails if they drift again.
- **Extras restructured.** No extra references the package itself. `everything`
  is an explicit flat union, verified by test against the union of all other
  extras — the check that would have caught the missing Phoenix and LangSmith
  declarations at commit time rather than at demo time.
  - `torch` → `dl` · `tree-models` → `trees` · `optuna` → `tuning`
  - `delta` + `snowflake` → `connectors` · `databricks`/`mlflow` → `tracking`
  - `openai`/`anthropic`/`llm-openai`/`llm-anthropic` → `llm`
  - `huggingface` + `genai` → `llm-local` · `ai-engineering` removed (name the
    backends you want)
  - `torch` and `all` retained as flat deprecated aliases so older docs keep working
- **`scripts/install_dependencies.py` removed.** `scripts/bootstrap.py` is the
  single installer; `bootstrap.sh` is now a thin wrapper with no logic of its own.
- **`requirements.txt` is generated.** Regenerate with
  `python scripts/sync_requirements.py`; CI fails if it is stale.
- **Top-level imports are lazy.** `import start` no longer pulls in pydantic,
  pandas or scikit-learn. Named imports still work; `from start import *` no
  longer eagerly resolves everything.

## Added

### Runtime profiles — one clone, two environments
`start.runtime_profile` decides what may be reached, and refuses at the routing
boundary rather than relying on configuration hygiene. `public_demo` /
`enterprise` / `airgapped`. Public SaaS providers are structurally unreachable
under `enterprise`, whatever SDKs are installed and whatever keys are exported.
The public-egress override exists and is recorded in every seal produced while
it is set.

### Gateways without repository edits
`START_GATEWAY_BASE_URL` for any OpenAI-compatible gateway, or a
`start.llm_gateways` entry point for anything else. Previously integration meant
editing `src/start/enterprise/llm_gateway.py`, which guaranteed a merge conflict
on every upstream pull. Credentials are referenced by *variable name*, so an
organisation's naming convention stays out of the repository.

### Risk core — `start.risk` (standard library only)
15 stripes, 13 object kinds, 22 dimensions, deterministic hash-stable plan
synthesis, and control-framework coverage against SR 11-7, BCBS 239, FRTB,
IFRS 9/CECL, ECOA/Reg B, EU AI Act, NIST AI RMF, ISO/IEC 42001, Interagency TPRM.

The central mechanism is **burden conservation**: an inapplicable dimension
transfers its obligation to named substitutes, which become mandatory. Nothing
is silently skipped, and "N/A" is no longer a place findings can hide.

### Attestation layer — `start.attestation` (standard library only)
- **Narrative invariance** — every figure in generated prose bound to evidence;
  invention and corrupted transcription block sealing, omission and scope
  difference do not. Honest rewording, including percent/proportion rendering,
  passes.
- **Disclosure envelopes** — prompts assembled only from a policy-derived
  projection, with an egress check on the actual outbound bytes.
- **Ledger replay** — chain verification that localises the break, plus
  cross-run metric comparison with volatile fields excluded.
- **Review seals** — ordered Merkle commitment reducing a whole review to one
  verifiable string, with per-leaf tamper localisation and inclusion proofs.

### CLI
`start risk stripes|objects|dimensions|plan|coverage` and
`start attest egress|replay|verify-seal`. Also `python -m start.risk`, which
needs nothing but Python.

### Demo and recording
`scripts/demo_flight.py` runs the full arc in about two minutes with or without
API keys, with deterministic timing for re-recording.
`scripts/record_demo.sh` produces the README assets.

## Fixed

- `scripts/bootstrap.py` contained a **syntax error** (`f".[{,.join(extras)}]"`)
  and could never have run. The "canonical installer" was not executable.
- The `observability` extra declared Langfuse but not LangSmith or Phoenix while
  the codebase shipped adapters for all three, so a "full" install left an
  adapter reporting `not_installed` permanently.
- `tests/test_enterprise_adapter.py` **hard-coded the organisation name** inside
  the assertion checking for its absence — putting the string in the repository,
  the diff and every clone. Now read from `START_FORBIDDEN_ORG_TOKEN`, set
  locally and never committed.
- `start/__init__.py` eagerly imported pydantic, so nothing imported in a
  locked-down environment.
- Claim extraction dropped any number followed by a full stop — silently missing
  every figure at the end of a sentence.
- Disclosure field-path globs did not match across list indices, so thresholds
  under `thresholds[0].warn` were silently withheld from every envelope.
- Merkle inclusion proofs mis-tracked the index across odd-node promotion:
  proofs verified for even-sized trees and failed for odd ones.

## Hardened

`.gitignore` now covers every common environment name, all `.env*` except
`.env.example`, outputs, caches and build artefacts. CI adds a job that fails if
a credential, tracked environment or generated artefact appears in the tree, and
pins `START_PROFILE=airgapped` so no test can reach a public endpoint.
