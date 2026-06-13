"""Isolated enterprise integration point for StART.

This package is the ONLY place a firm needs to modify to route StART's
agent-review prompts through a private internal LLM gateway. The public
repository never depends on the private package, never hardcodes endpoints,
URLs, or credentials, and degrades to deterministic review when no private
implementation is present.

See ``llm_gateway.py`` for the adapter and ``README.md`` for the integration
contract.
"""

from start.enterprise.llm_gateway import EnterpriseLLMGatewayAdapter

__all__ = ["EnterpriseLLMGatewayAdapter"]
