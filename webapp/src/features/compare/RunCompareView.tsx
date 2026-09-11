import { useEffect, useState } from 'react'
import type { RunCompareResult, RunHistoryItem } from '../../contracts/types'
import type { StartBackend } from '../../contracts/backend'
import { ReviewDialog } from '../../components/ReviewDialog'
import { DataBlock, DataValue, Empty, Section, ScientificTable } from '../investigation/Science'

export function ScientificComparison({ result }: {result: RunCompareResult}) {
  if (!result.compatible) return <Empty title="Runs are incompatible">{result.incompatibleReason || 'No compatibility reason supplied by the backend.'}</Empty>
  return <><div className="comparison-identities"><div><span>BASELINE</span><strong>{result.runA?.runId}</strong><p>{result.runA?.contextId}</p></div><div><span>CANDIDATE</span><strong>{result.runB?.runId}</strong><p>{result.runB?.contextId}</p></div></div>
    <DataBlock title="Configuration" data={result.parameters}/>
    <Section title="Metrics" note="Deltas supplied by the backend">{result.metricComparisons?.map(c => <details className="comparison-test" key={c.testId} open={c.isChanged}><summary><strong>{c.testId}</strong><span>{c.statusA} → {c.statusB} · {c.isChanged ? 'Changed' : 'Unchanged'}</span></summary><ScientificTable rows={c.metrics.map(m => ({metric:m.metric,baseline:m.valA,candidate:m.valB,delta:m.delta,percent_change:m.pctChange}))}/></details>)}<DataValue value={{only_in_baseline:result.onlyInA,only_in_candidate:result.onlyInB}}/></Section>
    <DataBlock title="Findings" data={result.findings}/><DataBlock title="Governance" data={result.governance}/><DataBlock title="Artifacts" data={result.artifacts}/><DataBlock title="Lineage" data={result.lineage}/>
  </>
}
export function RunCompareView(p: {isOpen:boolean;onClose:()=>void;runAId:string|null;runBId:string|null;compareResult:RunCompareResult|null;loading:boolean;onSelectRunA:(id:string)=>void;onSelectRunB:(id:string)=>void;onExecuteCompare:(a:string,b:string)=>void;onLoadRun:(id:string)=>void;backend:StartBackend}) {
  const [runs,setRuns] = useState<RunHistoryItem[]>([]),[error,setError] = useState('')
  useEffect(() => {if(!p.isOpen)return;let alive=true;(p.backend.listRuns?.() ?? Promise.resolve([])).then(r => {if(alive)setRuns(r)}).catch(e => {if(alive)setError(e.message)});return()=>{alive=false}},[p.isOpen,p.backend])
  if(!p.isOpen)return null
  const currentResult = p.compareResult && (!p.compareResult.runA || p.compareResult.runA.runId === p.runAId) && (!p.compareResult.runB || p.compareResult.runB.runId === p.runBId) ? p.compareResult : null
  return <ReviewDialog title="Compare runs" wide onClose={p.onClose}><div className="comparison-controls">{(['A','B'] as const).map(side => <label key={side}>{side === 'A' ? 'Baseline' : 'Candidate'}<select value={(side==='A'?p.runAId:p.runBId) ?? ''} onChange={e => (side==='A'?p.onSelectRunA:p.onSelectRunB)(e.target.value)}><option value="">Select a run</option>{runs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.workflow}</option>)}</select></label>)}<button className="tonal" disabled={!p.runAId || !p.runBId || p.loading} onClick={() => p.onExecuteCompare(p.runAId!,p.runBId!)}>Compare</button></div>{error && <p role="alert">{error}</p>}{p.loading ? <p>Loading scientific comparison…</p> : currentResult ? <ScientificComparison result={currentResult}/> : <Empty title="Select two reviews to compare"/>}</ReviewDialog>
}
