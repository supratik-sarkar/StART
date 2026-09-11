import '../features/certification/certification.css'
import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { DemoBackend } from '../adapters/demo/DemoBackend'
import { PublicStARTBackend } from '../adapters/public/PublicStARTBackend'
import { demoReviewerRuntime } from '../adapters/demo/DemoReviewerRuntime'
import { PublicReviewer } from '../adapters/public/PublicReviewer'
import { useWorkbench } from '../state/useWorkbench'
import type { ArtifactRecord, EdaProfile, EvidenceRecord, ExecutionMode, ProposedAction, RunLineage, ScenarioItem, StARTCapabilityManifest, TestCatalogItem } from '../contracts/types'
import { WorkbenchHeader } from '../components/WorkbenchHeader'
import { ReviewDialog } from '../components/ReviewDialog'
import { ResizableWorkspace } from '../components/ResizableWorkspace'
import { runConfiguration } from '../features/capabilities/runtimeSupport'
import { Composer } from '../features/composer/Composer'
import { EdaCanvas } from '../features/eda/EdaCanvas'
import { ExecutionReview } from '../features/investigation/ExecutionReview'
import { InvestigationWorkbench } from '../features/investigation/InvestigationWorkbench'
import { fromArtifacts, isAnalyticalObject, label, useCanonicalPresentation } from '../features/investigation/presentation'
import { DataValue, Empty, Section } from '../features/investigation/Science'
import { TypedArtifactRenderer } from '../features/artifacts/TypedArtifactRenderer'
import { RunHistoryModal } from '../features/history/RunHistoryModal'
import { RunCompareView } from '../features/compare/RunCompareView'
import { CommandPalette } from '../features/search/CommandPalette'
import { IterationModal } from '../features/execution/IterationModal'

const CertificationSurface = lazy(() => import('../features/certification/CertificationSurface').then(m => ({default:m.CertificationSurface})))

const mode = import.meta.env.VITE_START_ADAPTER || (import.meta.env.PROD ? 'public' : 'demo')
if (import.meta.env.PROD && mode === 'demo' && import.meta.env.VITE_ENABLE_DEMO !== 'true') throw new Error('PRODUCTION_DEMO_BACKEND = DISABLED')
const adapter = mode === 'public' ? new PublicStARTBackend(import.meta.env.VITE_START_API_BASE || '') : new DemoBackend()
const reviewer = mode === 'public' ? new PublicReviewer(adapter as PublicStARTBackend) : demoReviewerRuntime

export default function App() {
  const w = useWorkbench(adapter, reviewer)
  const [scienceVisited,setScienceVisited]=useState(false)
  const [surface,setSurface]=useState<'workbench'|'data'|'certification'>('workbench')
  const [section,setSection] = useState('Overview'), [requestedArtifact,setRequestedArtifact] = useState<string|null>(null)
  const [scenarios,setScenarios] = useState<ScenarioItem[]>([]), [tests,setTests] = useState<TestCatalogItem[]>([])
  const [scenario,setScenario] = useState<ScenarioItem|null>(null), [eda,setEda] = useState<EdaProfile|null>(null), [edaLoading,setEdaLoading] = useState(false), [edaError,setEdaError] = useState<string|null>(null)
  const [dialog,setDialog] = useState<'catalog'|'pins'|'runtime'|'ask'|'decision'|null>(null), [action,setAction] = useState<ProposedAction|null>(null)
  const [runtime,setRuntime] = useState(reviewer.getState()), [runtimeLabel,setRuntimeLabel] = useState(''), [localBusy,setLocalBusy] = useState(false)
  const [decision,setDecision] = useState<'ACCEPT'|'CHALLENGE'|'OVERRIDE'|'ESCALATE'>('ACCEPT'), [rationale,setRationale] = useState('')
  const [lineage,setLineage] = useState<RunLineage|null>(null)
  const [parameters,setParameters]=useState<Record<string,any>>({})
  const [seed,setSeed]=useState<number|undefined>(), [trials,setTrials]=useState<number|undefined>()
  const [plannedConfiguration,setPlannedConfiguration]=useState<string|null>(null)
  const [manifestError,setManifestError]=useState('')
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('hybrid_workbench')
  const [profileMode, setProfileMode] = useState<'workbench' | 'enterprise'>('workbench')
  const [manifest, setManifest] = useState<StARTCapabilityManifest | null>(null)
  const canonicalState = useCanonicalPresentation(w.run,adapter.adapterMode==='public')
  const canonical = canonicalState?.result ?? fromArtifacts(w.artifacts)
  const requestId=useRef(0)
  useEffect(()=>{(adapter.listScenarios?.() ?? Promise.resolve([])).then(setScenarios).catch(e=>w.setError(e.message));(adapter.listTests?.() ?? Promise.resolve([])).then(setTests).catch(e=>w.setError(e.message));adapter.getCapabilityManifest?.().then(setManifest).catch(()=>setManifestError('Capability manifest could not be loaded. Provider and scientific availability are unverified.'));return reviewer.subscribeState?.((state,progress)=>{setRuntime(state);if(progress)setRuntimeLabel(progress.label)})},[])
  useEffect(()=>{setSection('Overview');setRequestedArtifact(null);setLineage(null);if(w.run?.runId&&adapter.getRunLineage){let alive=true;adapter.getRunLineage(w.run.runId).then(r=>{if(alive)setLineage(r)}).catch(()=>{});return()=>{alive=false}}},[w.run?.runId])
  useEffect(()=>{if(w.run?.phase==='completed'){setSection('Overview');setRequestedArtifact(null)}},[w.run?.phase])
  useEffect(()=>{
    const id=++requestId.current
    setEda(null);setEdaError(null)
    if(w.run || !w.selectedContext){setEdaLoading(false);return}
    const source=scenario && scenario.compatible_context_id===w.selectedContext && adapter.getScenarioEda ? adapter.getScenarioEda(scenario.id) : adapter.getContextEda?.(w.selectedContext)
    if(!source)return
    setEdaLoading(true);source.then(p=>{if(requestId.current===id)setEda(p)}).catch(e=>{if(requestId.current===id)setEdaError(e.message)}).finally(()=>{if(requestId.current===id)setEdaLoading(false)})
    return()=>{requestId.current++}
  },[w.selectedContext,scenario?.id,!!w.run])
  const fresh=()=>{setSurface('workbench');setParameters({});setSeed(undefined);setTrials(undefined);setPlannedConfiguration(null);setScenario(null);setEda(null);setSection('Overview');setRequestedArtifact(null);w.reset()}
  useEffect(()=>{if(w.run?.runId)setSurface('workbench')},[w.run?.runId])
  const openRun=async(id:string)=>{setSurface('workbench');await w.loadRun(id)}
  const showEvidence=(id:string)=>{w.setSelectedEvidenceId(id || null);setRequestedArtifact(null);setSection('Evidence')}
  const pin=(record:ArtifactRecord|EvidenceRecord)=>{const artifact='artifactId' in record;const id=artifact?record.artifactId:record.evidenceId;if(w.isPinned(id))w.unpinItem(id);else w.pinItem({itemId:id,itemType:artifact?'artifact':'evidence',runId:record.runId,label:artifact?record.label:record.title})}
  const openQuestion=(id?:string)=>{if(id)w.setSelectedEvidenceId(id);setRationale('');setDialog('ask')}
  const openChallenge=(id?:string)=>{if(id)w.setSelectedEvidenceId(id);setRationale('Challenge this finding. Determine whether the stated conclusion is fully supported by the cited EvidenceRecord and its provenance. Identify any limitation, ambiguity, or unsupported interpretation without introducing new numerical calculations.');setDialog('ask')}
  const openDecision=(type:typeof decision,id?:string)=>{if(id)w.setSelectedEvidenceId(id);setDecision(type);setRationale('');setDialog('decision')}
  const isInvestigation=!!w.run && (w.run.phase==='completed'||w.isReplay)
  const artifacts=w.artifacts.filter(a=>a.runId===w.run?.runId && isAnalyticalObject(a))
  const activeArtifact=artifacts.find(a=>a.artifactId===requestedArtifact) ?? artifacts[0]
  const profile=eda?.data_profile ?? eda
  const hasEda=!!(profile && (profile.schema?.length || profile.feature_moments?.length || profile.correlation?.matrix?.length || profile.target_distribution || profile.overview && Object.keys(profile.overview).length))
  const initReviewer=async()=>{setLocalBusy(true);try{await reviewer.initialize(p=>{setRuntime(p.state);setRuntimeLabel(p.label)});setRuntime(reviewer.getState())}catch(e){setRuntime('unavailable');setRuntimeLabel((e as Error).message)}finally{setLocalBusy(false)}}
  const configurationKey=JSON.stringify([w.selectedWorkflow,w.selectedContext,w.goal,seed,trials,executionMode,parameters])
  const preparePlan=async()=>{try{const config=runConfiguration(w.selectedWorkflow,seed,trials,{...parameters,dataset_id:w.selectedContext},executionMode);await w.previewPlan({...config,contextId:w.selectedContext!});setPlannedConfiguration(configurationKey)}catch(e){w.setError((e as Error).message)}}
  const executePlan=()=>{if(plannedConfiguration!==configurationKey)return;try{w.startRun({...runConfiguration(w.selectedWorkflow,seed,trials,{...parameters,dataset_id:w.selectedContext},executionMode),contextId:w.selectedContext!})}catch(e){w.setError((e as Error).message)}}
  const aiReady=adapter.adapterMode==='public'?manifest?.ai_provider?.status==='ONLINE':runtime==='ready'
  const eventExecutionMode=w.events.find(e=>e.metadata?.execution_mode)?.metadata?.execution_mode
  const shownExecutionMode=w.run&&['hybrid_workbench','agentic_session','deterministic_run'].includes(String(eventExecutionMode))?eventExecutionMode as ExecutionMode:executionMode
  const header=<WorkbenchHeader mode={!w.run?'DEFINE':isInvestigation?'INVESTIGATE':'EXECUTE'} context={w.run ? label(w.run.contextId) : undefined} isReplay={w.isReplay} runId={w.run?.runId} executionMode={shownExecutionMode} onExecutionModeChange={!w.run?setExecutionMode:undefined} profileMode={profileMode} onProfileModeToggle={()=>setProfileMode(p=>p==='workbench'?'enterprise':'workbench')} onNew={fresh} onHistory={w.openHistory} onCompare={()=>w.openCompare()} onSearch={w.openSearch} onPins={()=>setDialog('pins')} onRuntime={()=>setDialog('runtime')} runtime={manifest?.ai_provider?.status==='ONLINE'?'Ready':manifest?.ai_provider?.status==='OFFLINE'?'Offline':'Unverified'} aiModel={manifest?.ai_provider?.model ?? 'Unverified'} aiProvider={manifest?.ai_provider?.provider ?? 'Unverified'}/>
  return <div className="start-workstation">
    {header}{manifestError&&<p className="workspace-alert" role="status">{manifestError}</p>}{w.error&&<div className="workspace-alert" role="alert">{w.error}<button className="text-action" onClick={()=>w.setError(null)}>Dismiss</button></div>}
    {canonicalState?.error&&isInvestigation&&<p className="workspace-alert">{canonicalState.error}. Available run artifacts remain inspectable.</p>}
    <nav className="workbench-surface-nav" aria-label="Workspace surfaces">{([['workbench','Workspace'],['data','Data Runtime'],['certification','Scientific Certification']] as const).map(([id,name])=><button key={id} aria-current={surface===id?'page':undefined} onClick={()=>{setSurface(id);if(id!=='workbench')setScienceVisited(true)}}>{name}</button>)}<span>Evidence retains its source</span></nav>
    <div hidden={surface==='workbench'}>{scienceVisited&&<Suspense fallback={<p role="status">Loading scientific workspace…</p>}><CertificationSurface surface={surface==='workbench'?'data':surface} context={w.selectedContext??undefined} onReturn={()=>setSurface('workbench')}/></Suspense>}</div>
    {surface!=='workbench'?null:!w.run ? <><ResizableWorkspace isRightPaneUnlocked={hasEda} ariaLabelLeft={profileMode==='enterprise'?'Define review':'Define workspace'} ariaLabelRight="Dataset inspection" leftContent={<Composer capabilities={w.capabilities} contexts={w.contexts} scenarios={scenarios} selectedScenario={scenario} onSelectScenario={s=>{setScenario(s);w.setSelectedContext(s.compatible_context_id)}} workflow={w.selectedWorkflow} setWorkflow={v=>{setParameters({});setTrials(undefined);setScenario(null);w.setSelectedWorkflow(v)}} context={w.selectedContext} setContext={v=>{setScenario(null);w.setSelectedContext(v)}} goal={w.goal} setGoal={w.setGoal} parameters={parameters} onParametersChange={setParameters} seed={seed} onSeedChange={setSeed} trials={trials} onTrialsChange={setTrials} plan={plannedConfiguration===configurationKey?w.plan:null} onPlan={preparePlan} onStart={executePlan} busy={w.busy} adapterMode={adapter.adapterMode} onOpenTestCatalog={()=>setDialog('catalog')} executionMode={executionMode} onExecutionModeChange={setExecutionMode} profileMode={profileMode} manifest={manifest}/>} rightContent={<EdaCanvas profile={eda} activeScenario={scenario} loading={edaLoading} error={edaError} onOpenTestCatalog={()=>setDialog('catalog')}/>}/>{!hasEda&&(edaLoading||edaError)&&<div className="profile-status" role="status">{edaError ?? 'Inspecting selected dataset…'}</div>}</> : isInvestigation ? <><div className="review-actions"><span>{w.isReplay ? 'Read-only reconstruction' : profileMode==='enterprise' ? 'Review complete' : 'Run complete'} · <code>{w.run.runId}</code></span><div><button className="text-action" onClick={fresh}>{profileMode==='enterprise'?'New review':'New workspace'}</button><button className="tonal" onClick={()=>w.openCompare(w.run!.runId)}>Compare</button>{!w.isReplay&&<button className="text-action" onClick={()=>setAction({actionId:crypto.randomUUID(),kind:'rerun',label:'Create child run',description:'',sourceEvidenceId:w.selectedEvidenceId ?? undefined,sourceNodeId:w.selectedNodeId ?? undefined,parameters:{}})}>Rerun</button>}</div></div><InvestigationWorkbench run={w.run} canonical={canonical} artifacts={w.artifacts} evidence={w.evidence} findings={w.findings} governance={w.governance} attestation={w.attestation} events={w.events} isReplay={w.isReplay} section={section} onSection={setSection} selectedEvidenceId={w.selectedEvidenceId} onEvidence={showEvidence} selectedArtifactId={requestedArtifact} onArtifact={setRequestedArtifact} onAsk={w.isReplay||executionMode==='deterministic_run'?undefined:openQuestion} onChallenge={w.isReplay||executionMode==='deterministic_run'?undefined:openChallenge} onDecision={()=>openDecision('ACCEPT')} onPin={pin} isPinned={w.isPinned} lineage={lineage} onLoadRun={openRun}/></> : <ResizableWorkspace isRightPaneUnlocked={!!activeArtifact} ariaLabelLeft="Execution review" ariaLabelRight="Live analytical output" leftContent={<ExecutionReview run={w.run} events={w.events} evidence={w.evidence} checkpoints={w.checkpoints} onEvidence={id=>{w.setSelectedEvidenceId(id);openQuestion(id)}}/>} rightContent={activeArtifact&&<div className="analytical-inspector"><div className="eyebrow">ANALYTICAL OUTPUT</div><h2>{activeArtifact.label}</h2><select aria-label="Select analytical output" value={activeArtifact.artifactId} onChange={e=>setRequestedArtifact(e.target.value)}>{artifacts.map(a=><option key={a.artifactId} value={a.artifactId}>{a.label}</option>)}</select><TypedArtifactRenderer artifact={activeArtifact}/></div>}/>}
    <footer className="workstation-footer"><span>StART · Agentic AI Engineering Workbench</span><span>{adapter.adapterMode==='demo'?'DEMO · Preview engine':'Connected backend'} · Evidence retains its source</span></footer>
    <RunHistoryModal isOpen={w.isHistoryOpen} onClose={w.closeHistory} onOpenRun={openRun} onCompareRun={w.openCompare} backend={adapter}/>
    <RunCompareView isOpen={w.isCompareOpen} onClose={w.closeCompare} runAId={w.compareRunA} runBId={w.compareRunB} compareResult={w.compareResult} loading={w.compareLoading} onSelectRunA={w.setCompareRunA} onSelectRunB={w.setCompareRunB} onExecuteCompare={w.executeCompare} onLoadRun={openRun} backend={adapter}/>
    <CommandPalette isOpen={w.isSearchOpen} onClose={w.closeSearch} onOpenRun={openRun} onSelectEvidence={async(id,run)=>{if(run&&run!==w.run?.runId)await w.loadRun(run);showEvidence(id)}} onSelectArtifact={async(id,run)=>{if(run&&run!==w.run?.runId)await w.loadRun(run);setSection('Artifacts');setRequestedArtifact(id)}} onSelectTest={id=>{const e=w.evidence.find(e=>e.testId===id);if(e)showEvidence(e.evidenceId);else setDialog('catalog')}} onSelectContext={id=>{fresh();w.setSelectedContext(id)}} pinnedItems={w.pinnedItems} backend={adapter}/>
    {dialog==='catalog'&&<ReviewDialog title={`Test catalog · ${tests.length} tests`} onClose={()=>setDialog(null)} wide><Catalog tests={tests}/></ReviewDialog>}
    {dialog==='pins'&&<ReviewDialog title="Pinned records" onClose={()=>setDialog(null)}>{w.pinnedItems.length ? w.pinnedItems.map(p=><article className="pinned-row" key={p.id}><button className="text-action" onClick={async()=>{if(p.runId!==w.run?.runId)await w.loadRun(p.runId);if(p.itemType==='evidence')showEvidence(p.itemId);else if(p.itemType==='artifact'){setSection('Artifacts');setRequestedArtifact(p.itemId)}else setSection('Findings');setDialog(null)}}>{p.label}</button><small>{p.runId}</small><button className="text-action" onClick={()=>w.unpinItem(p.itemId)}>Unpin</button></article>):<Empty title="No pinned records">Pin evidence or artifacts to keep them close.</Empty>}</ReviewDialog>}
    {dialog==='runtime'&&<ReviewDialog title="AI provider" onClose={()=>setDialog(null)}><p>Evidence-grounded questions are handled by the backend. No provider credentials enter the browser.</p><DataValue value={manifest?.ai_provider ?? {status:'Unverified'}}/>{adapter.adapterMode==='demo'&&<button className="tonal" disabled={localBusy||runtime==='ready'} onClick={initReviewer}>Initialize demo runtime</button>}<p className="quiet">Plan proposals and session checkpoints are supplied by the backend; they do not imply autonomous multi-agent deliberation.</p></ReviewDialog>}
    {dialog==='ask'&&<ReviewDialog title="Ask about this evidence" onClose={()=>setDialog(null)}><p>{w.selectedEvidence?.title ?? 'Current run'}</p><DataValue value={w.selectedEvidence?.metrics}/><label className="stacked-label">Question<textarea value={rationale} onChange={e=>setRationale(e.target.value)}/></label><button className="tonal" disabled={!rationale.trim()||localBusy||!aiReady} onClick={async()=>{setLocalBusy(true);try{await w.askAgent(rationale);setRationale('')}finally{setLocalBusy(false)}}}>Ask Workbench Agent</button>{!aiReady&&<p className="quiet">AI provider unavailable. <button className="text-action" onClick={()=>setDialog('runtime')}>View runtime</button></p>}<div className="contextual-answers">{w.messages.map(m=><article key={m.id}><span className="eyebrow">{m.role}</span><p>{m.text}</p></article>)}</div></ReviewDialog>}
    {dialog==='decision'&&<ReviewDialog title="Human review decision" onClose={()=>setDialog(null)}><DataValue value={{run:w.run?.runId,evidence:w.selectedEvidence?.title,source_evidence:w.selectedEvidenceId}}/><label className="stacked-label">Action<select value={decision} onChange={e=>setDecision(e.target.value as typeof decision)}><option value="ACCEPT">Record Approval</option><option value="CHALLENGE">Challenge</option><option value="OVERRIDE">Override Request</option><option value="ESCALATE">Escalate</option></select></label><label className="stacked-label">Rationale<textarea value={rationale} onChange={e=>setRationale(e.target.value)}/></label><button className={decision==='ACCEPT'?'primary':'tonal'} disabled={!rationale.trim()||localBusy||w.isReplay} onClick={async()=>{setLocalBusy(true);try{const receipt=await w.recordDecision(decision,rationale,w.selectedNodeId??undefined,undefined,w.selectedEvidenceId?[w.selectedEvidenceId]:[]);if(receipt)setDialog(null);else w.setError('Decision recording is unavailable in this backend.')}catch{}finally{setLocalBusy(false)}}}>{localBusy?'Recording…':decision==='ACCEPT'?'Record Approval':'Record decision'}</button><Section title="Decision receipts"><DataValue value={w.decisions}/></Section></ReviewDialog>}
    {action&&w.run&&!w.isReplay&&<IterationModal action={action} parentRunId={w.run.runId} currentParameters={canonical.resolved_configuration} onApprove={a=>{setAction(null);w.executeAction(a)}} onReject={()=>setAction(null)}/>}
  </div>
}
function Catalog({tests}:{tests:TestCatalogItem[]}){const [query,setQuery]=useState('');return <><input aria-label="Filter catalog" placeholder="Search tests, families, and descriptions" value={query} onChange={e=>setQuery(e.target.value)}/>{tests.filter(t=>`${t.name} ${t.testId} ${t.family} ${t.description}`.toLowerCase().includes(query.toLowerCase())).map(t=><article className="catalog-row" key={t.testId}><h3>{t.name}</h3><p>{t.description}</p><small>{t.family} · {t.testId}</small></article>)}</>}
