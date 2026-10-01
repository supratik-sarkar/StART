import type { AttestationState, DecisionReceipt, EvidenceRecord, GovernanceState, RunSnapshot, RuntimeEvent } from '../../contracts/types'
import { CopyValue } from './Science'

const absent = <span className="outcome-unavailable">Unavailable from this run</span>

export function evidenceStatusSummary(evidence: EvidenceRecord[]) {
  return ['FAIL', 'ERROR', 'WARN', 'ATTENTION', 'PASS', 'RECORDED', 'INFORMATIONAL', 'SKIPPED', 'NOT_APPLICABLE'].map(status => ({
    status,
    records: evidence.filter(record => record.status === status),
  })).filter(group => group.records.length > 0)
}

export function runDecisionReceipts(runId: string, decisions: DecisionReceipt[], events: RuntimeEvent[]): DecisionReceipt[] {
  const records = [...decisions, ...events.filter(event => event.runId === runId && event.type === 'human_decision' && typeof event.metadata?.receipt_id === 'string').map(event => event.metadata as unknown as DecisionReceipt)]
  return [...new Map(records.filter(record => (record.run_id ?? record.runId) === runId).map(record => [record.receipt_id ?? record.receiptId, record])).values()]
}

export function RunOutcome({run, evidence, governance, attestation, decisions, events, onNavigate}: {
  run: RunSnapshot
  evidence: EvidenceRecord[]
  governance: GovernanceState | null
  attestation: AttestationState | null
  decisions: DecisionReceipt[]
  events: RuntimeEvent[]
  onNavigate: (section: string, evidenceId?: string) => void
}) {
  const statuses = evidenceStatusSummary(evidence)
  const receipts = runDecisionReceipts(run.runId, decisions, events)
  const agentTraces = events.filter(event => event.runId === run.runId && /agent_decision/i.test(event.type) && event.metadata)
  const conditions = governance?.unresolvedItems ?? []
  const executed = events.filter(event => ['tool_completed', 'test_completed', 'evidence_created'].includes(event.type))
  return <section className="run-outcome" aria-labelledby="run-outcome-title">
    <header className="outcome-heading"><div><span className="eyebrow">COMPLETED RUN / REVIEW SUMMARY</span><h2 id="run-outcome-title">What this run established</h2></div><CopyValue value={run.runId} truncate/></header>
    <div className="outcome-grid">
      <div><h3>Started with</h3><p>{run.contextId || absent}</p><small>{run.workflowId || 'Workflow unavailable'}</small></div>
      <div><h3>Objective</h3><p>{run.goal || absent}</p></div>
      <div><h3>Alternatives</h3>{absent}<small>No comparable options were supplied in the completed run record.</small></div>
      <div><h3>Human decisions</h3>{receipts.length ? <><strong>{receipts.length} recorded</strong><button className="text-action" onClick={() => onNavigate('Provenance')}>Inspect receipts</button></> : absent}</div>
      <div><h3>Agent contribution</h3>{agentTraces.length ? <><strong>{agentTraces.length} structured decision events</strong><button className="text-action" onClick={() => onNavigate('Provenance')}>Inspect trace fields</button></> : absent}<small>{agentTraces.length ? 'Only supplied decision fields are displayed.' : 'No structured decision trace was supplied for this run.'}</small></div>
      <div><h3>Deterministic execution</h3>{executed.length ? <><strong>{executed.length} supplied completion events</strong><button className="text-action" onClick={() => onNavigate('Provenance')}>Inspect execution trace</button></> : absent}<small>Numerical results come from recorded execution and evidence, not agent text.</small></div>
      <div className="outcome-evidence"><h3>Evidence</h3>{statuses.length ? <div className="outcome-statuses">{statuses.map(group => <span key={group.status} className={`scientific-status status-${group.status.toLowerCase()}`}>{group.records.length} {group.status}</span>)}</div> : absent}<button className="text-action" onClick={() => onNavigate('Evidence')}>Inspect evidence ledger</button>{statuses.find(group => group.status === 'FAIL')?.records.map(record => <button key={record.evidenceId} className="outcome-failure text-action" onClick={() => onNavigate('Evidence', record.evidenceId)}>FAIL · {record.testId} · {record.evidenceId}</button>)}</div>
      <div><h3>Grounding</h3>{absent}<small>No run-level claim grounding result was supplied.</small></div>
      <div><h3>Governance</h3><strong className={governance?.disposition?.includes('CONDITIONS') ? 'outcome-conditional' : undefined}>{governance?.disposition ?? absent}</strong><button className="text-action" onClick={() => onNavigate('Governance')}>Inspect disposition</button></div>
      <div><h3>Policy result</h3><strong>{governance?.policyDecision ?? absent}</strong><small>A policy decision is separate from governance disposition.</small></div>
      <div><h3>Achieved result</h3>{absent}<small>Execution completion and attestation do not establish objective achievement.</small></div>
      <div className="outcome-conditions"><h3>Remaining conditions</h3>{conditions.length ? <ul>{conditions.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul> : governance?.disposition?.includes('CONDITIONS') ? <p>Conditions apply; their details were not supplied in the governance record.</p> : governance?.disposition?.includes('PENDING_REVIEW') ? <p>A challenge is pending review; resolution requirements were not supplied.</p> : absent}</div>
      <div><h3>Run attestation</h3>{attestation ? <CopyValue value={attestation.merkleRoot} truncate/> : absent}<button className="text-action" onClick={() => onNavigate('Provenance')}>Inspect attestation</button></div>
    </div>
    <p className="outcome-boundary">Run evidence, attestation, certification bundles, and historical comparisons each retain their own source and scope.</p>
  </section>
}
