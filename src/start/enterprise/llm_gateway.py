"""Enterprise LLM gateway adapter (public placeholder).

This is the single file a firm implements to integrate a private internal LLM
gateway. In the public repository it is inert: it reports unavailable unless a
private package (referred to here generically as the "enterprise package") is
importable, and its ``generate`` raises ``NotImplementedError`` until the firm
supplies the real routing.

Hard rules honored by this file:
  * No hardcoded endpoints, URLs, or credentials.
  * The private package is NEVER imported at module load — only lazily, inside
    methods, so the public repo has zero dependency on it and importing this
    module can never fail.
  * The adapter exposes the SAME interface as the public providers:
        generate(prompt, *, system=None, metadata=None) -> str
  * It accepts only evidence-grounded prompts (assembled upstream from the
    evidence bundle); raw rows / raw data are never passed in by the framework.

To integrate inside the firm environment, implement ``generate`` below (or add
``firm_adapter.py`` and delegate to it) WITHOUT touching agents, providers,
modeling, notebooks, or examples.
"""

from __future__ import annotations

import importlib.util
from typing import Any

# Generic name of the private, firm-internal package. The public repo does not
# ship, vendor, or depend on this package; it is only probed for at runtime.
# A firm may override this via the START_ENTERPRISE_PACKAGE environment
# variable if their internal package has a different import name.
ENTERPRISE_PACKAGE = "enterprise_package"


def _enterprise_package_name() -> str:
    import os

    return os.environ.get("START_ENTERPRISE_PACKAGE", ENTERPRISE_PACKAGE)


def enterprise_package_available() -> bool:
    """True only if the private enterprise package is importable. Uses
    importlib spec lookup so nothing is actually imported (and no import error
    can ever surface) when the package is absent."""
    try:
        return importlib.util.find_spec(_enterprise_package_name()) is not None
    except (ImportError, ValueError):
        return False


# Metadata keys the adapter understands. Passing extras is harmless; the
# framework supplies these so a firm implementation can route, throttle, or
# tag requests without any change to the calling code.
SUPPORTED_METADATA = (
    "run_id",
    "agent_name",
    "section_name",
    "max_tokens",
    "temperature",
    "model_name",
    "evidence_ids",
)


class EnterpriseLLMGatewayAdapter:
    """Public placeholder adapter for a firm-internal LLM gateway.

    Same interface as the public providers. Unavailable (and ``generate``
    raises) until a private implementation is supplied inside the firm
    environment — at which point selecting ``enterprise_llm_gateway`` works
    across agents, notebooks, terminal demos, and the evidence critic with no
    other changes.
    """

    provider_name = "enterprise_llm_gateway"

    def available(self) -> bool:
        return enterprise_package_available()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Route an evidence-grounded prompt through the enterprise gateway.

        Public placeholder: raises until implemented inside the firm
        environment. A firm implementation should call its internal gateway
        here using only ``prompt``/``system`` (already evidence-grounded) and
        the advisory ``metadata`` keys in ``SUPPORTED_METADATA``. It must never
        receive or forward raw dataset rows.
        """
        if not self.available():
            raise NotImplementedError(
                "Enterprise LLM gateway is unavailable: the private enterprise "
                f"package ('{_enterprise_package_name()}') is not importable in "
                "this environment. StART falls back to deterministic review."
            )
        raise NotImplementedError(
            "Enterprise LLM gateway is a private adapter. Implement this method "
            "inside the firm environment (edit src/start/enterprise/llm_gateway.py "
            "or add src/start/enterprise/firm_adapter.py and delegate to it). "
            "The framework passes only evidence-grounded prompts and the metadata "
            f"keys: {', '.join(SUPPORTED_METADATA)}."
        )
