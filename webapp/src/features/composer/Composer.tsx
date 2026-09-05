import { ArrowRight, BrainCircuit, ChartNoAxesCombined, Database, GitCompareArrows, Network, ScanSearch, SlidersHorizontal, Sparkles, Target, WandSparkles } from 'lucide-react'
import type { AgentPlanPreview, Capability, ExecutionContext, WorkflowId } from '../../contracts/types'

const icons:any={predictive_ml:BrainCircuit,deep_learning:Network,data_diagnostics:Database,model_diagnostics:ScanSearch,calibration:Target,robustness:SlidersHorizontal,explainability:Sparkles,hyperparameter_tuning:WandSparkles,model_comparison:GitCompareArrows,quantitative_finance:ChartNoAxesCombined}
export function Composer({capabilities,contexts,workflow,setWorkflow,context,setContext,goal,setGoal,plan,onPlan,onStart,busy,adapterMode}:{capabilities:Capability[];contexts:ExecutionContext[];workflow:WorkflowId|null;setWorkflow:(v:WorkflowId)=>void;context:string|null;setContext:(v:string)=>void;goal:string;setGoal:(v:string)=>void;plan:AgentPlanPreview|null;onPlan:()=>void;onStart:()=>void;busy:boolean;adapterMode:string}){
 const activeContext=contexts.find(c=>c.id===context)
 return <main className="composer-shell">
  <section className="composer-hero">
   <div className="eyebrow"><span className="eyebrow-dot"/> Evidence-native execution</div>
   <h1>What do you want StART<br/>to work on?</h1>
   <p>Describe the engineering goal, choose a real execution context, and inspect the plan before any deterministic work begins.</p>
   <div className="prompt-box">
    <textarea value={goal} onChange={e=>setGoal(e.target.value)} placeholder="Evaluate a classifier for calibration and robustness, then show me the evidence path…" />
    <div className="prompt-footer"><span>{adapterMode==='demo'?'Preview adapter · deterministic simulated events':'Connected backend'}</span><button disabled={!workflow||!context||busy} onClick={plan?onStart:onPlan}>{busy?'Preparing…':plan?'Run StART':'Build agent plan'} <ArrowRight size={16}/></button></div>
   </div>
  </section>
  <section className="workflow-section">
   <div className="section-heading"><div><span>01</span><h2>Choose the work</h2></div><p>ML/DL first. Quantitative workflows are available without defining the product.</p></div>
   <div className="workflow-grid">{capabilities.map(c=>{const I=icons[c.id]||Sparkles;return <button key={c.id} className={`workflow-card ${workflow===c.id?'selected':''} ${!c.enabled?'disabled':''}`} disabled={!c.enabled} onClick={()=>setWorkflow(c.id)} title={c.disabledReason}><div className="workflow-icon"><I size={18}/></div><div><strong>{c.label}</strong><p>{!c.enabled&&c.disabledReason?c.disabledReason:c.description}</p></div><ArrowRight className="workflow-arrow" size={16}/></button>})}</div>
  </section>
  <section className="context-section">
   <div className="section-heading"><div><span>02</span><h2>Choose execution context</h2></div><p>The context is visible before analysis; nothing is precomputed on load.</p></div>
   <div className="context-grid">{contexts.map(c=><button key={c.id} onClick={()=>setContext(c.id)} className={`context-card ${context===c.id?'selected':''}`}><div className="context-top"><div className="context-glyph"><Database size={17}/></div><span>{c.kind}</span></div><strong>{c.label}</strong><p>{c.description}</p><div className="context-meta"><span>{c.shape||'versioned context'}</span><span>{c.target?`target · ${c.target}`:`seed · ${c.seed}`}</span></div><div className="badge-row">{c.badges?.map(b=><em key={b}>{b}</em>)}</div></button>)}</div>
  </section>
  {plan&&<section className="plan-preview"><div className="plan-head"><div><span className="eyebrow">03 · Proposed execution</span><h2>Agent plan</h2><p>Visible before execution. Each step maps to a backend capability or runtime object.</p></div><button className="primary" onClick={onStart}>Execute plan <ArrowRight size={16}/></button></div><div className="plan-track">{plan.plan.map((s,i)=><div className="plan-step" key={s.id}><div className="plan-index">{String(i+1).padStart(2,'0')}</div><div><strong>{s.label}</strong><span>{s.kind}</span></div>{i<plan.plan.length-1&&<div className="plan-line"/>}</div>)}</div>{activeContext&&<div className="plan-context"><Database size={15}/><span>{activeContext.label}</span><i/> <span>{workflow?.replaceAll('_',' ')}</span></div>}</section>}
 </main>
}
