# Backend Gap Checklist for Antigravity

Before changing backend code, inspect existing StART APIs and mark each row **EXISTS / THIN-ENDPOINT-NEEDED / NOT-SUPPORTED**.

- [ ] capability registry endpoint
- [ ] execution context / dataset catalog endpoint
- [ ] agent plan preview endpoint
- [ ] create run endpoint
- [ ] run snapshot endpoint
- [ ] ordered/reconnectable event stream
- [ ] progress fields from real work units
- [ ] execution graph / parent-child lineage endpoint
- [ ] EvidenceRecord list/detail endpoint
- [ ] Findings endpoint or presentation-model mapping
- [ ] Artifact list/detail endpoint
- [ ] contextual agent message endpoint
- [ ] explicit action / child-run endpoint
- [ ] governance endpoint
- [ ] attestation endpoint
- [ ] reviewer submission → server hydration/gating endpoint

Only create thin transport/presentation surfaces for gaps. Do not duplicate deterministic analytics in the web layer.
