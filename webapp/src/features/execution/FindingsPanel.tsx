import { AlertTriangle, ArrowRight, BrainCircuit, MessageCircleMore, SearchCheck } from 'lucide-react'
import type { Finding } from '../../contracts/types'

export function FindingsPanel({findings,onInspectEvidence,onAsk}:{findings:Finding[];onInspectEvidence:(id:string)=>void;onAsk:(text:string)=>void}){
  return <section className="findings-panel">
    <div className="findings-intro"><span className="eyebrow">Evidence-backed findings</span><h3>{findings.length ? `${findings.length} item${findings.length===1?'':'s'} to inspect` : 'No evidence-derived attention items yet'}</h3><p>Findings are presentation objects over canonical evidence. They never replace the underlying EvidenceRecords.</p></div>
    <div className="findings-list">{findings.map(f=><article className={`finding-card severity-${f.severity||'info'}`} key={f.findingId}>
      <div className="finding-icon">{f.severity==='attention'?<AlertTriangle size={17}/>:<BrainCircuit size={17}/>}</div>
      <div className="finding-body"><div className="finding-top"><span className="mono">{f.findingId}</span><span>{f.severity||'info'}</span></div><h4>{f.title}</h4><p>{f.summary}</p>
      <div className="finding-evidence">{f.evidenceIds.map(e=><button key={e} onClick={()=>onInspectEvidence(e)}><SearchCheck size={12}/>{e}</button>)}</div>
      {f.limitations?.length?<div className="finding-limit"><strong>Limitation</strong><span>{f.limitations[0]}</span></div>:null}
      <div className="finding-actions"><button onClick={()=>onAsk(`Explain finding ${f.findingId} using only its evidence.`)}><MessageCircleMore size={13}/> Explain</button>{f.availableActions.includes('challenge')&&<button onClick={()=>onAsk(`Challenge finding ${f.findingId}. If a deterministic follow-up is warranted, propose it explicitly.`)}>Challenge</button>}{f.availableActions.includes('deeper_test')&&<button className="strong" onClick={()=>onAsk(`Run a deeper deterministic test for finding ${f.findingId}. Show me the proposed action before execution.`)}>Deeper test <ArrowRight size={13}/></button>}</div></div>
    </article>)}</div>
  </section>
}
