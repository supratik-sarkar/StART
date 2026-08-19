# Enterprise integration

How to run StART inside an organisation that mandates an internal inference
gateway, **without editing a single file in this repository**.

That constraint is the whole design goal. An integration that requires editing
tracked files creates a permanent merge conflict: every upstream pull collides
with the local change, and eventually someone stops pulling. Both routes below
leave the repository untouched.

---

## Route 1 — configuration only (OpenAI-compatible gateways)

Most internal gateways speak the OpenAI chat-completions wire format, because
that is what every client library already talks. If yours does, you need no code.

```bash
export START_GATEWAY_BASE_URL=https://<host>/<path>/v1
export START_GATEWAY_API_KEY_ENV=MY_ORG_TOKEN     # the NAME of the variable
export START_GATEWAY_MODEL=<model name>

start doctor
start review --provider gateway
```

| Variable | Meaning |
|---|---|
| `START_GATEWAY_BASE_URL` | Base URL. Required — StART will not guess an address. |
| `START_GATEWAY_API_KEY_ENV` | Name of the environment variable holding the token. Defaults to `START_GATEWAY_API_KEY`. |
| `START_GATEWAY_MODEL` | Model identifier. Required — StART will not guess a model name on an operator-supplied gateway. |
| `START_GATEWAY_EXTRA_HEADERS` | JSON object of additional headers (routing tags, cost centres). |
| `START_GATEWAY_TIMEOUT` | Request timeout in seconds. Default 60. |

### Why the credential is passed by variable name

StART is told *which variable holds the token*, never the token, and never a
fixed variable name.

An organisation's credential naming convention is itself an internal detail. If
StART hard-coded `ACME_LLM_TOKEN`, that string would live in this public
repository, in every clone, and in the diff history permanently. The indirection
costs one extra export and keeps the identifier where it belongs.

The same reasoning applies to headers: `gateway_settings()` returns header
*keys* for diagnostics and withholds the values, because a header is a common
place to put a token.

### Authentication by network position

Some gateways authenticate by mTLS or by network position and need no bearer
token. StART treats a reachable base URL as sufficient and reports the
credential state in `start doctor` rather than refusing to start.

---

## Route 2 — entry point (everything else)

For a gateway with a bespoke protocol, request signing, or a client library of
its own, ship a **private wheel** installed alongside StART.

### The contract

```python
# my_private_pkg/start_gateway.py

class Gateway:
    """StART gateway adapter. Lives in your private package, not in StART."""

    def available(self) -> bool:
        """True when this adapter can serve a request right now."""
        return _credentials_present() and _endpoint_reachable()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Route an evidence-grounded prompt and return the completion text.

        `prompt` and `system` are already constrained by the disclosure
        envelope (see below). `metadata` is advisory and may carry:
        run_id, agent_name, section_name, max_tokens, temperature,
        model_name, evidence_ids.
        """
        return _your_client.complete(system=system, user=prompt, **_map(metadata))
```

### Registration

```toml
# my_private_pkg/pyproject.toml

[project.entry-points."start.llm_gateways"]
my_gateway = "my_private_pkg.start_gateway:Gateway"
```

The target may be a class or a zero-argument factory. Then:

```bash
pip install ./my-private-pkg
start doctor                        # lists my_gateway as registered
start review --provider my_gateway
```

### What StART does and does not do

- Discovery is by entry-point **metadata only** — nothing is imported until the
  gateway is selected. A private package that fails to import cannot prevent
  StART from starting or from reporting an accurate runtime profile.
- The public repository never imports, vendors, names or depends on your package.
- Agents, evidence, attestation and sealing are unaware of which implementation
  answered. Nothing downstream changes.

---

## What the profile guarantees

Installing a gateway (either route) makes StART infer the `enterprise` runtime
profile. Under it, every public SaaS provider is **refused at the routing
boundary**:

```console
$ start review --provider openai
ProfileViolation: Provider 'openai' reaches a third-party inference endpoint,
which the 'enterprise' runtime profile does not permit.
```

This holds regardless of installed SDKs or exported credentials. It is a
structural refusal, not a check for missing configuration.

Pin it explicitly in a managed environment rather than relying on inference:

```bash
export START_PROFILE=enterprise      # or airgapped for no outbound inference at all
```

### The escape hatch, and why it is loud

`START_ALLOW_PUBLIC_EGRESS=true` re-admits public providers under the
`enterprise` profile. It exists because a platform team may need to A/B a public
model during onboarding.

Using it is recorded in the profile manifest, which is a sealed leaf of every
review produced while it is set. The manifest hash differs from an uncontained
one, so a reviewer can tell — later, from the archive alone — whether a given
review ran under full containment. The hatch is available; it is not quiet.

---

## Disclosure policy

The gateway receives prompts assembled from a **policy-derived projection** of
the evidence, not from the evidence itself.

| Policy | Projects | Default for |
|---|---|---|
| `public_demo` | metrics, thresholds, statuses, interpretations, identifiers | `public_demo` profile |
| `restricted` | metrics, thresholds, test identity and status only | `enterprise` profile |
| `minimal` | test identity and pass/warn/fail only — no magnitudes | `airgapped` profile |

Under `restricted`, a narrative can describe what was measured without
reproducing what it was measured on. Under `minimal`, quantitative content in
the report comes entirely from the deterministic path; the model can say what
failed, never by how much.

Two enforcement points:

1. **Construction.** Only allow-listed field paths are projected. A *denied*
   path (raw rows, identifiers, free-text notes) present in the input aborts
   envelope construction rather than being dropped — its presence means an
   upstream caller assembled something it should not have, and silently
   filtering would leave that bug in place.
2. **Egress.** Every numeric token in the outbound prompt must exist in the
   envelope, checked immediately before the call, when the actual bytes are known.

Override per run when a review genuinely needs different bounds:

```python
from start.attestation import build_envelope, policy_for

envelope = build_envelope(evidence, policy=policy_for(override="minimal"))
```

The override is recorded in the envelope metadata and therefore in the seal.

---

## Deployment checklist

- [ ] `START_PROFILE=enterprise` (or `airgapped`) pinned in the environment, not inferred
- [ ] Gateway configured by Route 1 or Route 2; `start doctor` reports it available
- [ ] `START_ALLOW_PUBLIC_EGRESS` **unset**
- [ ] Disclosure policy confirmed appropriate — `start attest egress`
- [ ] `start review --provider openai` fails with `ProfileViolation` (verify the refusal actually fires)
- [ ] Seal manifests archived somewhere durable; `start attest verify-seal` runs against them
- [ ] Ledger files retained for `start attest replay`
- [ ] `.env` in `.gitignore`; no credential in any tracked file
- [ ] `START_FORBIDDEN_ORG_TOKEN` set locally so `tests/test_enterprise_adapter.py` scans for your organisation's name — in your local `.env` only, never committed

---

## Verifying an archived review later

Verification needs Python and nothing else — no StART installation, no network,
no original environment:

```bash
python -m start.risk egress                      # what regime did it run under
start attest replay archive/ledger.jsonl
start attest verify-seal archive/manifest.json --seal start-seal/1:R-2026-0042:e740943e...
```

If the seal verifies, the plan, policy, evidence head, attestations, containment
regime and control coverage are all exactly what they were at sign-off. If it
does not, the failing leaf is named.
