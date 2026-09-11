export type RecordData = Record<string, any>
export interface CertificationBundle {
  schema_version: number; packaged_at: string
  sources: {file:string; sha256:string; bytes:number}[]
  runtime: RecordData; certification_manifest: RecordData
  dataset_manifest: RecordData[]; experiment_matrix: RecordData[]
  deterministic_runs: RecordData[]; gpt41_runs: RecordData[]
  champion_challenger: RecordData[]; invariant_results: RecordData[]
  xai_results: RecordData[]; sensitivity_results: RecordData[]
  provider_trace: RecordData[]; failures: RecordData[]
  certification_discrepancies: RecordData
  real_data_domain_coverage: RecordData[]; gpt41_policy_coverage: RecordData[]
}
export const domains: Record<string,string> = {
  predictive_binary:'Predictive', deep_learning:'Deep Learning', fraud_imbalanced:'Fraud / AML',
  recommender:'Recommender', portfolio:'Portfolio', market_risk:'Market Risk', scenario_traded_risk:'Scenario / Traded Risk',
}
export const providers: Record<string,string> = {huggingface:'Hugging Face',openml:'OpenML',uci:'UCI',kaggle:'Kaggle',local_csv:'Local CSV',local_parquet:'Local Parquet'}
export const layerLabel = (layer: string) => layer === 'real_external' ? 'REAL EXTERNAL' : layer === 'golden_known_answer' ? 'GOLDEN / KNOWN-ANSWER' : 'Layer not supplied'
export function validateBundle(value: any): CertificationBundle {
  if (value?.schema_version !== 1 || !value.certification_manifest?.certification_id || !value.certification_manifest?.status_dimensions) throw new Error('Certification bundle schema is unavailable or unsupported.')
  for (const key of ['sources','dataset_manifest','experiment_matrix','deterministic_runs','gpt41_runs','champion_challenger','invariant_results','xai_results','sensitivity_results','provider_trace','failures','real_data_domain_coverage','gpt41_policy_coverage']) {
    if (!Array.isArray(value[key])) throw new Error(`Certification bundle is missing ${key}.`)
  }
  return value
}
export async function loadBundle(signal: AbortSignal): Promise<CertificationBundle> {
  const root = `${import.meta.env.BASE_URL}certification/`
  const response = await fetch(root+'index.json',{signal,cache:'no-store'})
  if (!response.ok) throw new Error('Certification snapshot is unavailable. Export the existing bundle and rebuild the frontend.')
  const index = await response.json()
  if (!/^[a-f0-9]{64}\.json$/.test(index.file) || index.file !== `${index.sha256}.json`) throw new Error('Invalid certification snapshot reference.')
  const artifact = await fetch(root+index.file,{signal})
  if (!artifact.ok) throw new Error('Certification artifact could not be loaded.')
  const bytes = await artifact.arrayBuffer()
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('')
  if (hash !== index.sha256) throw new Error('Certification snapshot integrity check failed.')
  const bundle = validateBundle(JSON.parse(new TextDecoder().decode(bytes)))
  if (bundle.certification_manifest.certification_id !== index.certification_id) throw new Error('Certification identity mismatch.')
  return bundle
}
export interface ComparisonRow {model:string; metric:string; mean:number; std?:number; ci95?:number[]; policy:string; designation:string; task?:string}
// Field mapping only. The backend owns all ranks, statistics, and confidence bounds.
export function comparisonRows(record: RecordData): ComparisonRow[] {
  if (record.domain === 'portfolio' || record.domain === 'market_risk' || record.domain === 'scenario_traded_risk') return []
  if (record.architectures_certified) return Object.entries(record.architectures_certified).flatMap(([model, raw]) => {
    const value = raw as RecordData, metricKey = Object.keys(value).find(k => k.startsWith('mean_'))
    return metricKey ? [{model,metric:metricKey.slice(5),mean:value[metricKey],std:value.std,ci95:value.ci95,policy:'deterministic',designation:'Architecture result',task:value.task}] : []
  })
  if (!record.champion_model || typeof record.champion_mean !== 'number') return []
  return [{model:record.champion_model,metric:record.primary_metric,mean:record.champion_mean,std:record.champion_std,ci95:record.champion_ci95,policy:record.champion_policy,designation:'Reported champion'},
    ...(record.challenger_summaries ?? []).map((r:RecordData)=>({model:r.model,metric:record.primary_metric,mean:r.mean,std:r.std,ci95:r.ci95,policy:r.policy,designation:'Challenger'}))]
}
export function experimentRuns(bundle:CertificationBundle, experiment:string) {
  return [...bundle.deterministic_runs,...bundle.gpt41_runs].filter(r=>r.experiment_id===experiment)
}
export function experimentTrace(bundle:CertificationBundle, experiment:string) {
  return bundle.provider_trace.find(t=>t.experiment_id===experiment)
}
// Join identities by the exact dataset/revision/fingerprint in emitted run records.
// DL sequence/vision identities are present in JSONL even when absent from dataset_manifest.
export function datasetIdentities(bundle:CertificationBundle, runs:RecordData[]):RecordData[] {
  const seen = new Set<string>()
  return runs.flatMap(run=>{
    const key=JSON.stringify([run.dataset,run.dataset_revision,run.dataset_fingerprint])
    if(seen.has(key))return [];seen.add(key)
    const manifest=bundle.dataset_manifest.find(d=>d.dataset_id===run.dataset&&d.revision===run.dataset_revision&&d.fingerprint===run.dataset_fingerprint)
    return [{...manifest,dataset_id:run.dataset,provider:run.provider,revision:run.dataset_revision,fingerprint:run.dataset_fingerprint,layer:run.layer,rows:manifest?.rows ?? null,features:manifest?.features ?? null,target:manifest?.target ?? null}]
  })
}
