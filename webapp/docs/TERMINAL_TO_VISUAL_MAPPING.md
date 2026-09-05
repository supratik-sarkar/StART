# Terminal → Visual Mapping

This is the canonical integration checklist for Antigravity. The browser should expose an equal or better representation of every meaningful terminal concept.

| Terminal / runtime concept | Contract | Primary visual |
|---|---|---|
| selected workflow/domain | `RunSnapshot.workflowId` | workbench run header + composer |
| public dataset / synthetic world | `ExecutionContext` | composer context card + context node |
| execution plan | `RunSnapshot.plan` | pre-run Agent Plan + Living Execution Path |
| parent operation | `ExecutionGraphNode.parentId` | parent/child tree + inspector breadcrumb |
| child branch | graph edge `branch` | animated branch in Lineage / execution path |
| tool call | node kind `tool`, `RuntimeEvent.tool_*` | structured tool node + Tool Inspector |
| registered deterministic test | node kind `test` | execution node + runtime event + evidence link |
| progress | `ProgressState` | persistent top progress strip + active path animation |
| evidence creation | `EvidenceRecord` / `evidence_created` | Evidence Explorer + evidence tag on source node |
| quantitative metric | `EvidenceRecord.metrics` | Evidence Inspector only; never recomputed in React |
| finding | `Finding` | Findings tab with evidence links/actions |
| agent explanation | `ConversationMessage` | contextual Agent ↔ Human surface |
| proposed deterministic action | `ProposedAction` | action card inside conversation before execution |
| human decision | message/action + graph node `human` | lineage node and conversation history |
| re-run / intervention | child `RunSnapshot.parentRunId` | rerun edge + child run lineage |
| artifact | `ArtifactRecord` | Artifacts tab / rich preview |
| governance | `GovernanceState` | final sign-off + governance node |
| attestation | `AttestationState` | sign-off seal + attestation inspector |
| failure / recoverable state | `RunPhase`, node status | exact node/path state + sanitized explanation |

## Integration invariant

Do not map terminal strings directly to UI copy if typed backend objects exist. Prefer canonical IDs, objects and event schemas so the browser remains robust when terminal formatting changes.
