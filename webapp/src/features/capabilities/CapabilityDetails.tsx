import type { StARTCapabilityManifest } from '../../contracts/types'
import { DataValue } from '../investigation/Science'
import { capabilityLabel } from './runtimeSupport'

export function CapabilityDetails({manifest}:{manifest:StARTCapabilityManifest|null|undefined}) {
  if (!manifest) return <p role="status">Capability manifest unavailable. Scientific options cannot be verified.</p>
  return <details className="capability-disclosure"><summary>Capability availability & configuration</summary><p className="field-help">The backend declares the following capabilities. Its current public executor does not expose their configuration schemas or apply these selections. These are read-only declarations, not executable choices.</p>
    {Object.entries(manifest.domains).filter(([id])=>!['dataset_hub','governance'].includes(id)).map(([id,domain])=><details key={id}><summary>{domain.name}</summary><p className="attention-note">{id==='llm_agents'?'Evidence questions and human decisions are connected. Automated committee execution has no public runtime route.':'Configuration unavailable in the current public runtime.'}</p>{Object.entries(domain).filter(([key])=>key!=='name'&&!['active_provider','active_model','strict_zero_substitution'].includes(key)).map(([key,value])=><section key={key}><h3>{capabilityLabel(key)}</h3>{Array.isArray(value)?<ul className="capability-values">{value.map((v,i)=><li key={i}>{typeof v==='string'?capabilityLabel(v):String(v)}</li>)}</ul>:key==='sensitivity_analysis'?<><p>Top global features: {String((value as any).max_features ?? 'Not supplied')}</p><div className="shock-grid" aria-label="Declared sensitivity shock grid">{((value as any).shocks_grid ?? []).map((v:number)=><span key={v}>{v>0?'+':''}{v*100}%{v===0?' · baseline':''}</span>)}</div><DataValue value={{modes:(value as any).modes,fixed_model:'No public configuration binding',retraining_support:(value as any).retraining_support}}/></>:<DataValue value={value}/>}</section>)}</details>)}
    <p className="quiet">Unavailable or deferred engines are excluded from execution. Model winners are established by supplied results.</p>
  </details>
}
