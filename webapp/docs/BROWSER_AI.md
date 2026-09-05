# Browser AI Runtime Contract

Browser AI is deliberately separate from `StartBackend` because it may run entirely inside the user's browser.

Contract: `src/contracts/reviewer.ts`.

Public StART integration should preserve the accepted architecture:

- WebGPU browser runtime
- `@mlc-ai/web-llm` compatible reviewer
- pinned `SmolLM2-1.7B-Instruct-q4f16_1-MLC` unless the backend project deliberately changes it
- asynchronous model initialization
- deterministic run launch never awaits model readiness
- EvidenceRecords are the review input
- generated numeric claims remain untrusted until server-side Evidence-ID validation/hydration

The conversation surface should combine:

1. backend/orchestrator explanations and proposed tool actions; and
2. browser-local qualitative reviewer output.

They must remain distinguishable in state even if they share the same visual conversation surface.
