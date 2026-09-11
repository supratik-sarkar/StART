import { useEffect, useState } from 'react'
import type { StartBackend } from '../../contracts/backend'
import type { RunHistoryItem } from '../../contracts/types'
import { ReviewDialog } from '../../components/ReviewDialog'
import { formatDateTime } from '../../utils/formatTimestamp'
import { label } from '../investigation/presentation'
import { Empty } from '../investigation/Science'
export function RunLedger({ runs, onOpenRun, onCompareRun }: { runs: RunHistoryItem[]; onOpenRun: (id: string) => void; onCompareRun: (id: string) => void }) {
  const [query,setQuery] = useState('')
  const [status,setStatus] = useState('all')
  const visible = runs.filter(r => `${r.run_id} ${r.workflow} ${r.context_id} ${r.goal}`.toLowerCase().includes(query.toLowerCase()) && (status==='all' || r.status===status))
  return <><div className="ledger-controls"><input aria-label="Search run history" type="search" placeholder="Search runs, workflows, or contexts" value={query} onChange={e => setQuery(e.target.value)}/><select aria-label="Filter run status" value={status} onChange={e => setStatus(e.target.value)}>{['all','completed','running','queued','failed'].map(s => <option key={s}>{s}</option>)}</select><span className="quiet">{visible.length} runs</span></div><div className="science-table-scroll"><table className="science-table run-ledger"><thead><tr>{['Run','Workflow','Context','Status','Completed','Evidence','Artifacts','Disposition','Parent',''].map((s,i) => <th key={i}>{s}</th>)}</tr></thead><tbody>{visible.map(r => <tr key={r.run_id}><td><button className="text-action" onClick={() => onOpenRun(r.run_id)}>{r.run_id}</button></td><td>{label(r.workflow)}</td><td>{label(r.context_id)}</td><td><span className={`scientific-status status-${r.status}`}>{r.status}</span></td><td>{formatDateTime(r.completed_at)}</td><td>{r.evidence_count}</td><td>{r.artifact_count}</td><td>{r.governance_disposition || 'Not supplied'}</td><td>{r.parent_run_id ? <button className="text-action" onClick={() => onOpenRun(r.parent_run_id!)}>{r.parent_run_id}</button> : '—'}</td><td><button className="text-action" onClick={() => onCompareRun(r.run_id)}>Compare</button></td></tr>)}</tbody></table></div>{!visible.length && <Empty title="No matching runs"/>}</>
}
export function RunHistoryModal(p: {isOpen: boolean; onClose: () => void; onOpenRun: (id: string) => void; onCompareRun: (a: string,b?: string) => void; backend: StartBackend}) {
  const [runs,setRuns] = useState<RunHistoryItem[]>([]), [error,setError] = useState(''), [loading,setLoading] = useState(false)
  useEffect(() => {if(!p.isOpen) return;let alive=true;setLoading(true);setError('');(p.backend.listRuns?.() ?? Promise.resolve([])).then(r => {if(alive)setRuns(r)}).catch(e => {if(alive)setError(e.message)}).finally(() => {if(alive)setLoading(false)});return () => {alive=false}},[p.isOpen,p.backend])
  if(!p.isOpen)return null
  return <ReviewDialog wide title="Run history" onClose={p.onClose}>{loading ? <p>Loading run ledger…</p> : error ? <p role="alert">{error}</p> : <RunLedger runs={runs} onOpenRun={id => {p.onOpenRun(id);p.onClose()}} onCompareRun={id => {p.onCompareRun(id);p.onClose()}}/>}</ReviewDialog>
}
