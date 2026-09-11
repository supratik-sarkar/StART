import { useEffect, useState } from 'react'
import { DataValue, Empty, ScientificTable, Section } from '../investigation/Science'
import { liveRequest } from './liveApi'
import type { RecordData } from './bundle'
export function isSelectedRunPresentation(data:RecordData,runId:string) {return data.requested_run_id===runId&&data.source_run_id===runId&&data.source_scope==='RUN'&&data.status!=='NOT_AVAILABLE_FOR_RUN'}
export type PresentationKind='tuning'|'xai'|'sensitivity'|'portfolio'|'scenario'
export function LiveRunPresentation({runId,kind}:{runId:string;kind:PresentationKind}) {
  const [data,setData]=useState<RecordData|null>(null),[error,setError]=useState('')
  const [persisted,setPersisted]=useState<RecordData|null>(null)
  useEffect(()=>{const c=new AbortController();setData(null);setPersisted(null);setError('');
    const read=async()=>{const d=await liveRequest(`/runs/${encodeURIComponent(runId)}/${kind}`,{signal:c.signal});if(c.signal.aborted)return;setData(d);
      if(d.requested_run_id===runId&&!d.certification_id){
        try {const response=await liveRequest(`/runs/${encodeURIComponent(runId)}/presentation`,{signal:c.signal});const science=response.presentation?.scientific_presentation;
          if(!c.signal.aborted&&science?.run_id===runId&&science[kind])setPersisted({run_id:science.run_id,workflow:science.workflow,context_id:science.context_id,seed:science.seed,observations:science[kind]});
        } catch { /* Optional persisted detail is never replaced with other evidence. */ }
      }
    };void read().catch(e=>{if(!c.signal.aborted)setError(e.message)});return()=>c.abort()},[runId,kind])

  const identity=data?Object.fromEntries(['requested_run_id','source_run_id','experiment_id','certification_id','dataset_id','model_id','source_scope','status'].map(k=>[k,data[k]??null])):null
  const scoped=data&&isSelectedRunPresentation(data,runId)
  return <Section title={`Live ${kind==='xai'?'XAI':kind} presentation`} note="Backend source identity is required before observations are shown for this run.">{error?<Empty title="Live presentation unavailable">{error}</Empty>:!data?<p role="status">Loading supplied presentation…</p>:<><DataValue value={identity}/>{scoped?<>{Array.isArray(data.trials)&&data.trials.length>0&&<ScientificTable rows={data.trials}/>}<DataValue value={Object.fromEntries(Object.entries(data).filter(([k])=>!Object.hasOwn(identity!,k)))}/></>:<Empty title="NOT_AVAILABLE_FOR_RUN">{data.status==='NOT_AVAILABLE_FOR_RUN'?data.reason??'The backend has no observations for this run.':'The supplied source does not prove selected-run identity. Certification-wide or other-run observations are not displayed here.'}</Empty>}{persisted&&<><h3>Persisted execution observations</h3><p className="quiet">Stored by the backend for this exact run; historical details are not reconstructed.</p><DataValue value={persisted}/></>}</>}</Section>

}
export function LiveSeedDetail({run}:{run:RecordData}) {
  const [open,setOpen]=useState(false),[record,setRecord]=useState<RecordData|null>(null),[error,setError]=useState('')
  useEffect(()=>{if(!open)return;const c=new AbortController();setError('');liveRequest(`/certification/runs/${encodeURIComponent(run.run_id)}`,{signal:c.signal}).then(d=>{if(!c.signal.aborted)setRecord(d)}).catch(e=>{if(!c.signal.aborted)setError(e.message)});return()=>c.abort()},[open,run.run_id])
  return <details className="cert-seed" onToggle={e=>setOpen(e.currentTarget.open)}><summary><span>Seed {run.seed} · {run.policy}</span><span>{run.primary_metric_name} {String(run.primary_metric_value)}</span></summary>{open&&(error?<Empty title="Seed detail unavailable">{error}</Empty>:record?<><DataValue value={record}/><LiveRunPresentation runId={record.run_id} kind="tuning"/>{record.domain==='portfolio'&&<LiveRunPresentation runId={record.run_id} kind="portfolio"/>}{record.domain==='scenario_traded_risk'&&<LiveRunPresentation runId={record.run_id} kind="scenario"/>}{record.domain==='predictive_binary'&&record.layer==='real_external'&&<><LiveRunPresentation runId={record.run_id} kind="xai"/><LiveRunPresentation runId={record.run_id} kind="sensitivity"/></>}</>:<p role="status">Loading seed detail from live API…</p>)}</details>
}
