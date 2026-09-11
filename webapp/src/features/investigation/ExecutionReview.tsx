import type { CheckpointRecord, EvidenceRecord, RunSnapshot, RuntimeEvent } from '../../contracts/types'
import { DataValue, Empty, Section } from './Science'
import { formatDateTime } from '../../utils/formatTimestamp'

export function ExecutionTrace({ events }: { events: RuntimeEvent[] }) {
  return <details className="execution-trace"><summary>Execution Trace · {events.length} events <span>Show trace</span></summary><ol>{events.map(e => <li key={e.eventId}><time>{formatDateTime(e.timestamp)}</time><div><strong>{e.title || e.type}</strong><p>{e.message}</p></div><span className="quiet">{e.status}</span></li>)}</ol></details>
}
export function ExecutionReview({ run, events, evidence, checkpoints, onEvidence }: { run: RunSnapshot; events: RuntimeEvent[]; evidence: EvidenceRecord[]; checkpoints: CheckpointRecord[]; onEvidence: (id: string) => void }) {
  const active = run.plan.find(s => s.status === 'running')
  return <div className="execution-review"><div className="eyebrow">EXECUTE / {run.phase.toUpperCase()}</div><h1>{active?.label ?? run.statusLabel}</h1><p className="lead">{run.progress?.detail ?? active?.description ?? run.goal}</p>
  <Section title="Verification Milestones" note="Status from the execution runtime"><ol className="verification-flow">{run.plan.map(s => <li key={s.id} data-status={s.status}><span className="milestone-mark">{s.status === 'completed' ? '✓' : s.status === 'running' ? '●' : s.status === 'failed' ? '×' : '○'}</span><div><strong>{s.label}</strong><small>{s.status}</small><details><summary>Why this stage?</summary><p>{s.description ?? 'No stage rationale supplied.'}</p></details></div></li>)}</ol></Section>
  <Section title="New evidence" note={`${evidence.length} records available`}>{evidence.length ? evidence.slice(-5).reverse().map(e => <button className="evidence-row" key={e.evidenceId} onClick={() => onEvidence(e.evidenceId)}><span><strong>{e.title}</strong><small>{e.testId}</small></span><span className={`scientific-status status-${e.status.toLowerCase()}`}>{e.status}</span></button>) : <Empty title="Awaiting evidence">Records appear as the runtime completes its checks.</Empty>}</Section>
  {checkpoints.length > 0 && <details className="checkpoint-disclosure"><summary>Verification checkpoints · {checkpoints.length}</summary><DataValue value={checkpoints.map(c => ({name:c.name,status:c.status,summary:c.summary}))}/></details>}
  <ExecutionTrace events={events}/></div>
}
