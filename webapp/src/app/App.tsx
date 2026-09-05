import { useMemo, useState } from 'react'
import { Activity, Box, ChevronLeft, Command, FileStack, GitFork, PanelRightOpen, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react'
import { DemoBackend } from '../adapters/demo/DemoBackend'
import { PublicStARTBackend } from '../adapters/public/PublicStARTBackend'
import { Brand } from '../components/Brand'
import { useWorkbench } from '../state/useWorkbench'
import { Composer } from '../features/composer/Composer'
import { ExecutionPath } from '../features/execution/ExecutionPath'
import { EventStream } from '../features/execution/EventStream'
import { LineageGraph } from '../features/lineage/LineageGraph'
import { EvidenceExplorer } from '../features/evidence/EvidenceExplorer'
import { EvidenceInspector } from '../features/evidence/EvidenceInspector'
import { AgentConversation } from '../features/conversation/AgentConversation'
import { ToolInspector } from '../features/execution/ToolInspector'
import { Signoff } from '../features/governance/Signoff'
import { FindingsPanel } from '../features/execution/FindingsPanel'
import { ArtifactsPanel } from '../features/artifacts/ArtifactsPanel'

import { webLLMReviewer } from '../adapters/public/WebLLMReviewer'

const isProd = import.meta.env.PROD
const adapterMode = import.meta.env.VITE_START_ADAPTER || (isProd ? 'public' : 'demo')

if (isProd && adapterMode === 'demo' && import.meta.env.VITE_ENABLE_DEMO !== 'true') {
  throw new Error('PRODUCTION_DEMO_BACKEND = DISABLED. Production deployment must use PublicStARTBackend.')
}

const adapter = adapterMode === 'public'
  ? new PublicStARTBackend(import.meta.env.VITE_START_API_BASE || '')
  : new DemoBackend()

export default function App(){
 const w=useWorkbench(adapter); const [centerTab,setCenterTab]=useState<'events'|'lineage'|'findings'>('events'); const [rightTab,setRightTab]=useState<'context'|'evidence'|'artifacts'|'agent'>('context');
 const [reviewerStatus,setReviewerStatus]=useState<string>('idle');
 const [reviewerBusy,setReviewerBusy]=useState(false);

 const initReviewer = async () => {
   setReviewerBusy(true);
   try {
     await webLLMReviewer.initialize((p)=>{
       setReviewerStatus(p.percent!=null?`${p.label} (${p.percent}%)`:p.label);
     });
   } catch (err: any) {
     setReviewerStatus(`Error: ${err.message||err}`);
   } finally {
     setReviewerBusy(false);
   }
 };

 const runReview = async () => {
   if(!w.run) return;
   setReviewerBusy(true);
   try {
     const reviewOutput = await webLLMReviewer.review({
       runId: w.run.runId,
       goal: w.run.goal,
       evidence: w.evidence,
       contextNodeId: w.selectedNodeId || undefined
     });
     if (adapter.submitReviewerOutput) {
       await adapter.submitReviewerOutput(w.run.runId, reviewOutput);
       const [freshFindings, freshGov] = await Promise.all([
         adapter.getFindings(w.run.runId),
         adapter.getGovernance(w.run.runId)
       ]);
       // Update state
       setCenterTab('findings');
     }
   } catch(e:any){
     //
   } finally {
     setReviewerBusy(false);
   }
 };
 const selectedNode=useMemo(()=>w.graph.nodes.find(n=>n.id===w.selectedNodeId),[w.graph,w.selectedNodeId]);
 const workflowLabel=w.capabilities.find(c=>c.id===w.run?.workflowId)?.label
 if(!w.run) return <div className="app-shell"><header className="topbar"><Brand/><div className="top-actions"><span className="status-badge"><i/> Engine interface ready</span><span className={`status-badge ${adapter.adapterMode==='demo'?'preview':''}`}><Sparkles size={13}/>{adapter.adapterName}</span></div></header>{w.error&&<div className="global-error">{w.error}</div>}<Composer capabilities={w.capabilities} contexts={w.contexts} workflow={w.selectedWorkflow} setWorkflow={w.setSelectedWorkflow} context={w.selectedContext} setContext={w.setSelectedContext} goal={w.goal} setGoal={w.setGoal} plan={w.plan} onPlan={w.previewPlan} onStart={()=>w.startRun()} busy={w.busy} adapterMode={adapter.adapterMode}/><footer className="minimal-footer">StART · Agents orchestrate. Deterministic engines compute. Evidence is the product.</footer></div>
 return <div className="workbench"><header className="workbench-top"><div className="run-brand"><button onClick={w.reset}><ChevronLeft size={16}/></button><Brand/><span className="run-divider"/><div className="run-ident"><strong>{workflowLabel||w.run.workflowId}</strong><span>{w.run.runId}</span></div></div><div className="workbench-status"><span className={`phase phase-${w.run.phase}`}><Activity size={13}/>{w.run.phase.replaceAll('_',' ')}</span><span className="status-badge"><i/>{adapter.adapterMode==='demo'?'Preview engine':'Engine ready'}</span><button className="icon-button" onClick={w.reset}><RotateCcw size={15}/></button></div></header>
  {w.error&&<div className="global-error">{w.error}</div>}
  <div className="progress-strip"><div className="progress-copy"><span>{w.run.progress?.label||w.run.statusLabel}</span><small>{w.run.progress?.detail}</small></div><div className="progress-track">{w.run.progress?.percent!=null?<><div style={{width:`${w.run.progress.percent}%`}}/><i style={{left:`calc(${w.run.progress.percent}% - 5px)`}}/></>:<span>phase progress only</span>}</div><div className="progress-numeric">{w.run.progress?.completed!=null&&w.run.progress?.total!=null?<><strong>{w.run.progress.completed}</strong><span>/ {w.run.progress.total}</span></>:w.run.progress?.percent!=null?<strong>{Math.round(w.run.progress.percent)}%</strong>:<span>running</span>}</div></div>
  <div className="workspace-grid">
   <aside className="left-workspace"><ExecutionPath plan={w.run.plan} graph={w.graph} progress={w.run.progress} selected={w.selectedNodeId} onSelect={w.setSelectedNodeId}/></aside>
   <main className="center-workspace"><div className="workspace-tabs"><button className={centerTab==='events'?'active':''} onClick={()=>setCenterTab('events')}><Command size={14}/> Execution</button><button className={centerTab==='lineage'?'active':''} onClick={()=>setCenterTab('lineage')}><GitFork size={14}/> Lineage</button><button className={centerTab==='findings'?'active':''} onClick={()=>setCenterTab('findings')}><ShieldCheck size={14}/> Findings {w.findings.length?`· ${w.findings.length}`:''}</button><span/><button onClick={()=>setRightTab('agent')}><PanelRightOpen size={14}/> Ask agent</button></div>{centerTab==='events'?<EventStream events={w.events} onSelect={w.setSelectedNodeId}/>:centerTab==='lineage'?<LineageGraph graph={w.graph} selected={w.selectedNodeId} onSelect={w.setSelectedNodeId}/>:<FindingsPanel findings={w.findings} onInspectEvidence={id=>{w.setSelectedEvidenceId(id);setRightTab('evidence')}} onAsk={text=>{setRightTab('agent');w.askAgent(text)}}/>} {w.run.phase==='completed'&&<Signoff run={w.run} evidence={w.evidence} findings={w.findings} governance={w.governance} attestation={w.attestation} onTrace={()=>setCenterTab('lineage')}/>}</main>
   <aside className="right-workspace"><div className="right-tabs"><button className={rightTab==='context'?'active':''} onClick={()=>setRightTab('context')}><Box size={14}/> Inspect</button><button className={rightTab==='evidence'?'active':''} onClick={()=>setRightTab('evidence')}><FileStack size={14}/> Evidence</button><button className={rightTab==='artifacts'?'active':''} onClick={()=>setRightTab('artifacts')}><FileStack size={14}/> Artifacts</button><button className={rightTab==='agent'?'active':''} onClick={()=>setRightTab('agent')}><Sparkles size={14}/> Agent</button></div>{rightTab==='context'?<ToolInspector node={selectedNode} events={w.events}/>:rightTab==='evidence'?<div className="right-evidence"><EvidenceExplorer evidence={w.evidence} selected={w.selectedEvidenceId} onSelect={w.setSelectedEvidenceId}/><EvidenceInspector evidence={w.selectedEvidence}/></div>:rightTab==='artifacts'?<ArtifactsPanel artifacts={w.artifacts}/>:<AgentConversation messages={w.messages} onAsk={w.askAgent} onAction={w.executeAction} contextLabel={selectedNode?.label} reviewerStatus={reviewerStatus} onInitReviewer={initReviewer} onRunReview={runReview} reviewerBusy={reviewerBusy}/>}</aside>
  </div>
  <div className="bottom-context"><span><ShieldCheck size={13}/> Evidence-first</span><span><GitFork size={13}/> Parent path preserved</span><span><Sparkles size={13}/> AI never owns numeric truth</span><span className="mono">{adapter.adapterMode.toUpperCase()} ADAPTER</span></div>
 </div>
}
