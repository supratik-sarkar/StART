import type { CertificationBundle, RecordData } from './bundle'
const base=(import.meta.env.VITE_START_API_BASE||'').replace(/\/$/,'')+'/api/v1'
export async function liveRequest(path:string, options:{method?:string;body?:unknown;signal?:AbortSignal;providerSessionId?:string}={}) {
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),45000)
  const abort=()=>controller.abort();options.signal?.addEventListener('abort',abort,{once:true})
  if(options.signal?.aborted)controller.abort()
  try {
    const response=await fetch(base+path,{method:options.method??'GET',signal:controller.signal,cache:'no-store',headers:{...(options.body?{'Content-Type':'application/json'}:{}),...(options.providerSessionId?{'X-Provider-Session-ID':options.providerSessionId}:{})},body:options.body?JSON.stringify(options.body):undefined})
    if(!response.ok)throw new Error(`Live API request failed (${response.status}). Check backend availability and the selected dataset configuration.`)
    const envelope=await response.json()
    if(envelope.success!==true||!envelope.data||typeof envelope.data!=='object')throw new Error('Live API returned an unsupported response envelope.')
    return envelope.data as RecordData
  } finally {clearTimeout(timer);options.signal?.removeEventListener('abort',abort)}
}
export function mergeDomainRecords(index:RecordData[],details:RecordData[]) {
  return index.map(row=>{
    const detail=details.find(d=>d.domain===row.domain&&d.experiment_id===row.experiment_id&&(!d.selected_experiment_id||d.selected_experiment_id===row.experiment_id)&&(!d.selected_data_layer||d.selected_data_layer===(row.data_layer??row.layer)))
    return detail?{...row,...detail}:{...row,presentation_limitation:'BACKEND_BLOCKER: domain detail does not expose this experiment; challenger aggregates and policy details are unavailable.'}
  })
}
export async function loadLiveCertification(signal:AbortSignal):Promise<CertificationBundle> {
  const paths=['/certification','/certification/domains','/certification/invariants','/certification/xai','/certification/sensitivity','/certification/provider-traces','/certification/exclusions','/data/providers','/data/runtime']
  const settled=await Promise.allSettled(paths.map(p=>liveRequest(p,{signal})))
  const failed=settled.find(r=>r.status==='rejected')
  if(failed?.status==='rejected')throw failed.reason
  const [summary,index,invariants,xai,sensitivity,traces,exclusions,registry,runtime]=settled.map(r=>(r as PromiseFulfilledResult<RecordData>).value)
  if(!summary.bundle_hash||!Array.isArray(index.domains)||!Object.keys(summary.status_dimensions??{}).length)throw new Error('Live certification identity or domain records are missing.')
  const scopes:RecordData[]=index.scopes??index.domains
  const domainResults=await Promise.allSettled(scopes.map(scope=>liveRequest('/certification/domains/'+encodeURIComponent(scope.domain)+'?experiment_id='+encodeURIComponent(scope.experiment_id)+'&data_layer='+encodeURIComponent(scope.data_layer??scope.layer),{signal})))
  const detailFailure=domainResults.find(r=>r.status==='rejected');if(detailFailure?.status==='rejected')throw detailFailure.reason
  const details=domainResults.map(r=>(r as PromiseFulfilledResult<RecordData>).value)
  const runs:RecordData[]=[]
  for(let offset=0;offset<10000;){
    const page=await liveRequest(`/certification/runs?limit=200&offset=${offset}`,{signal})
    if(!Array.isArray(page.runs)||typeof page.total_filtered!=='number')throw new Error('Unsupported certification pagination response.')
    runs.push(...page.runs);offset+=page.runs.length
    if(offset>=page.total_filtered)break
    if(!page.runs.length||offset>=10000)throw new Error('Certification pagination is incomplete.')
  }
  const end=await liveRequest('/certification',{signal})
  if(end.bundle_hash!==summary.bundle_hash)throw new Error('Certification changed while loading. Refresh to inspect a consistent version.')
  return {
    schema_version:1,packaged_at:new Date().toISOString(),sources:summary.source_artifact_hashes??[],
    certification_manifest:{...summary,timestamp:summary.generated_at},
    runtime:{scope:'Live backend API',providers:Object.fromEntries((registry.providers??[]).map((r:RecordData)=>[r.provider_id,{...r,name:r.provider_id,details:r.reason_if_unavailable}])),parallel:runtime},
    dataset_manifest:details.map(d=>d.dataset_metadata).filter(d=>d?.dataset_id),experiment_matrix:index.domains.map((d:RecordData)=>d.experiment_spec).filter(Boolean),
    deterministic_runs:runs.filter(r=>r.policy==='deterministic'),gpt41_runs:runs.filter(r=>r.policy==='gpt41'),
    champion_challenger:mergeDomainRecords(index.domains,details),invariant_results:invariants.invariants??[],
    xai_results:xai.xai_methods??[],sensitivity_results:sensitivity.evaluations??[],provider_trace:traces.provider_traces??[],
    failures:exclusions.failures??[],certification_discrepancies:{historical_discrepancies_audited:exclusions.historical_discrepancies_audited,summary:exclusions.discrepancies_summary},
    real_data_domain_coverage:details.map(d=>d.real_data_coverage).filter(Boolean),gpt41_policy_coverage:details.map(d=>d.gpt41_policy_coverage).filter(Boolean),
  }
}
