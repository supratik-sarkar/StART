# Enterprise LLM Gateway Adapter

This folder is the **single, isolated integration point** for routing StART's
agent-review prompts through a private, firm-internal LLM gateway. It exists so
the public repository stays open-source-safe and a firm can integrate its own
gateway without touching the rest of the framework.

## What lives here

| File | Purpose |
| --- | --- |
| `llm_gateway.py` | The public placeholder adapter (`EnterpriseLLMGatewayAdapter`). |
| `firm_adapter.py` *(optional, firm-added)* | A private implementation you may add inside the firm environment. |
| `__init__.py` | Exports the adapter. |

## Design guarantees (public repo)

- **No firm specifics.** No endpoints, URLs, credentials, or internal package
  names are hardcoded anywhere.
- **No dependency on the private package.** The private package (referred to
  generically as the *enterprise package*) is never imported at module load —
  only probed for lazily with `importlib.util.find_spec`. Importing this module
  can never fail, even when the package is absent.
- **Same interface as public providers:** `generate(prompt, *, system=None,
  metadata=None) -> str`.
- **Evidence only.** The framework passes evidence-grounded prompts assembled
  from the evidence bundle. Raw dataset rows are never passed to the adapter.
- **Safe default.** With no private package present, the adapter reports
  `available() == False`, selecting `enterprise_llm_gateway` is a no-op that
  degrades to deterministic review, and no key is required.

## Public mode

- Deterministic review is the default and needs no key.
- Optional LLM mode uses hidden OpenAI / Anthropic / Grok key prompts
  (`--prompt-for-key` in the terminal, `getpass` in notebooks).
- `enterprise_llm_gateway` reports unavailable unless the private package
  exists, then falls back to deterministic review.

## Enterprise mode (inside the firm environment)

1. Install or expose your private *enterprise package* on the Python path. If
   its import name differs from the default, set the environment variable
   `START_ENTERPRISE_PACKAGE` to that name.
2. Implement `EnterpriseLLMGatewayAdapter.generate` in `llm_gateway.py` (or add
   `firm_adapter.py` and delegate to it). Use only `prompt` / `system` (already
   evidence-grounded) and the advisory metadata keys: `run_id`, `agent_name`,
   `section_name`, `max_tokens`, `temperature`, `model_name`, `evidence_ids`.
   Never accept or forward raw data rows.
3. Select the provider:
   - terminal: `--llm-provider enterprise_llm_gateway`
   - notebook widget: `llm_provider = enterprise_llm_gateway`
   - config: `agent.llm_provider: enterprise_llm_gateway`

No OpenAI / Anthropic / Grok key is required in enterprise mode.

## What you must NOT need to touch

Integrating the firm gateway should require editing **only this folder**. The
following remain unchanged:

- `src/start/agents/`
- `src/start/providers/`
- `src/start/modeling/`
- `notebooks/`
- `examples/`

The `EvidenceCriticAgent` still gates every section: no claim enters a report
without a valid evidence citation, regardless of which provider produced it.
