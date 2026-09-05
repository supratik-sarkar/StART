import type { StartBackend, StreamSubscription } from '../../contracts/backend'
import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, EvidenceRecord,
  ExecutionContext, ExecutionGraph, Finding, GovernanceState, ProposedAction, RunRequest,
  RunSnapshot, RuntimeEvent, WorkflowId
} from '../../contracts/types'

const now = () => new Date().toISOString()
const wait = (ms:number) => new Promise(r => setTimeout(r, ms))
const id = (p:string) => `${p}-${Math.random().toString(36).slice(2, 8).toUpperCase()}`

const caps: Capability[] = [
  ['predictive_ml','Predictive ML','Evaluate supervised models through deterministic engineering surfaces.','ml'],
  ['deep_learning','Deep Learning','Inspect architecture, training dynamics, diagnostics and evidence.','ml'],
  ['data_diagnostics','Data Diagnostics','Understand quality, drift, integrity and feature structure.','ml'],
  ['model_diagnostics','Model Diagnostics','Trace errors, residual behaviour and model-specific diagnostics.','ml'],
  ['calibration','Calibration','Inspect probabilistic reliability and calibration behaviour.','ml'],
  ['robustness','Robustness','Stress model behaviour under deterministic perturbations.','ml'],
  ['explainability','Explainability','Inspect evidence-backed attribution and sensitivity artefacts.','ml'],
  ['hyperparameter_tuning','Tune a Model','Run bounded search with truthful trial-level progress.','ml'],
  ['model_comparison','Compare Models','Evaluate candidates under a shared deterministic protocol.','ml'],
  ['quantitative_finance','Quantitative Finance','Run market, portfolio, scenario and risk engineering workflows.','quant'],
].map(([id,label,description,category]) => ({id:id as WorkflowId,label,description,category:category as 'ml'|'quant',enabled:id!=='model_comparison',disabledReason:id==='model_comparison'?'Multi-model candidate comparison workflow requires multi-candidate input protocol not yet enabled in the canonical web review interface.':undefined}))

const contexts: ExecutionContext[] = [
  { id:'credit-synth-v1', label:'Synthetic Credit Classification', kind:'dataset', description:'Seeded public-safe binary classification context for engineering workflows.', provenance:'Built-in deterministic synthetic generator', shape:'12,000 × 31', target:'default_flag', seed:42, badges:['public-safe','seeded','binary'] },
  { id:'vision-synth-v1', label:'Synthetic Vision Embeddings', kind:'dataset', description:'Compact embedding classification context for DL-oriented diagnostics.', provenance:'Built-in deterministic synthetic generator', shape:'8,000 × 128', target:'class_id', seed:17, badges:['public-safe','deep-learning'] },
  { id:'market-synth-v1', label:'Synthetic Multi-Asset Market World', kind:'synthetic-world', description:'Seeded multi-asset scenario context for portfolio and market workflows.', provenance:'Built-in deterministic synthetic market generator', shape:'24 assets × 1,500 observations', seed:7, badges:['public-safe','quantitative'] },
]

const mkPlan = (workflow:WorkflowId) => {
  const map: Record<WorkflowId, string[]> = {
    predictive_ml:['Load execution context','Validate data contract','Build evaluation plan','Run deterministic model checks','Run calibration branch','Run robustness branch','Create evidence bundle','Agent evidence review','Governance & sign-off'],
    deep_learning:['Load execution context','Inspect architecture','Initialize training diagnostics','Observe epoch/batch path','Run robustness branch','Generate interpretability evidence','Agent evidence review','Governance & sign-off'],
    data_diagnostics:['Load execution context','Validate schema','Inspect missingness','Inspect drift & distribution','Inspect feature structure','Create evidence bundle','Agent evidence review','Sign-off'],
    model_diagnostics:['Load execution context','Resolve model context','Inspect error structure','Inspect residual behaviour','Inspect stability','Create evidence bundle','Agent evidence review','Sign-off'],
    calibration:['Load execution context','Resolve score semantics','Measure calibration','Inspect reliability structure','Compare calibration states','Create evidence bundle','Review & sign-off'],
    robustness:['Load execution context','Resolve perturbation plan','Execute stress cases','Compare degradation paths','Create evidence bundle','Review & sign-off'],
    explainability:['Load execution context','Resolve compatible explainers','Generate attribution evidence','Inspect local/global structure','Create artifacts','Review & sign-off'],
    hyperparameter_tuning:['Load execution context','Validate search space','Establish baseline','Execute bounded trials','Compare candidates','Create evidence bundle','Review & sign-off'],
    model_comparison:['Load execution context','Resolve candidate set','Establish shared protocol','Evaluate candidates','Compare evidence','Create decision bundle','Review & sign-off'],
    quantitative_finance:['Load market world','Validate portfolio context','Build analytical plan','Run scenario & stress branches','Run portfolio/risk checks','Create evidence bundle','Review & governance','Attestation'],
  }
  return map[workflow].map((label,i)=>({id:`step-${i+1}`,label,kind:(i===0?'context':i===2?'agent':i>4?'evidence':'test') as any,status:(i===0?'completed':'queued') as any,parentId:i?`step-${i}`:undefined}))
}

interface DemoRunState {
  snapshot: RunSnapshot
  events: RuntimeEvent[]
  evidence: EvidenceRecord[]
  findings: Finding[]
  graph: ExecutionGraph
  artifacts: ArtifactRecord[]
  governance: GovernanceState | null
  attestation: AttestationState | null
  listeners: Set<(e:RuntimeEvent)=>void>
  timers: number[]
}

export class DemoBackend implements StartBackend {
  readonly adapterName='Greenfield deterministic preview adapter'
  readonly adapterMode='demo' as const
  private runs = new Map<string,DemoRunState>()
  async getCapabilities(){ return caps }
  async listExecutionContexts(){ return contexts }
  async createPlan(request:RunRequest):Promise<AgentPlanPreview>{
    return {workflowId:request.workflowId,contextId:request.contextId,goal:request.goal,plan:mkPlan(request.workflowId)}
  }
  async createRun(request:RunRequest):Promise<RunSnapshot>{
    const runId=id('RUN')
    const plan=mkPlan(request.workflowId)
    const snapshot:RunSnapshot={runId,workflowId:request.workflowId,contextId:request.contextId,goal:request.goal,phase:'planning',statusLabel:'Agent plan accepted',startedAt:now(),updatedAt:now(),elapsedMs:0,progress:{label:'Preparing deterministic execution'},plan,parentRunId:request.parentRunId,sourceEvidenceId:request.sourceEvidenceId}
    const state:DemoRunState={snapshot,events:[],evidence:[],findings:[],graph:{nodes:[],edges:[]},artifacts:[],governance:null,attestation:null,listeners:new Set(),timers:[]}
    this.runs.set(runId,state)
    this.seedGraph(state, request)
    this.scheduleRun(state, request)
    return structuredClone(snapshot)
  }
  async getRun(runId:string){ return structuredClone(this.must(runId).snapshot) }
  streamRun(runId:string,onEvent:(e:RuntimeEvent)=>void,onError?:(e:Error)=>void):StreamSubscription{
    const state=this.must(runId); state.listeners.add(onEvent); state.events.forEach(onEvent)
    return { close:()=>state.listeners.delete(onEvent) }
  }
  async getExecutionGraph(runId:string){ return structuredClone(this.must(runId).graph) }
  async getEvidence(runId:string){ return structuredClone(this.must(runId).evidence) }
  async getFindings(runId:string){ return structuredClone(this.must(runId).findings) }
  async getArtifacts(runId:string){ return structuredClone(this.must(runId).artifacts) }
  async submitHumanAction(runId:string,action:ProposedAction):Promise<RunSnapshot>{
    const parent=this.must(runId)
    return this.createRun({workflowId:parent.snapshot.workflowId,contextId:parent.snapshot.contextId,goal:action.label,parameters:action.parameters||{},parentRunId:runId,sourceEvidenceId:action.sourceEvidenceId,intervention:action.kind})
  }
  async submitReviewerOutput(runId:string,review:any){
    const runState = this.must(runId)
    return {
      runId,
      modelName:'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
      hydratedFindings:review.findings||[],
      allGrounded:true,
      governanceDisposition: runState.governance?.disposition || '',
      attestationSealMerkleRoot:'preview:7fd2a8…'
    }
  }
  async getGovernance(runId:string){ return structuredClone(this.must(runId).governance) }
  async getAttestation(runId:string){ return structuredClone(this.must(runId).attestation) }
  private must(runId:string){ const s=this.runs.get(runId); if(!s) throw new Error(`Unknown run ${runId}`); return s }
  private emit(state:DemoRunState,event:Omit<RuntimeEvent,'eventId'|'sequence'|'timestamp'|'runId'>){
    const e:RuntimeEvent={...event,eventId:id('EVT'),sequence:state.events.length+1,timestamp:now(),runId:state.snapshot.runId}
    state.events.push(e); state.snapshot.updatedAt=e.timestamp; state.snapshot.progress=e.progress||state.snapshot.progress
    state.listeners.forEach(l=>l(structuredClone(e)))
  }
  private seedGraph(state:DemoRunState, request:RunRequest){
    const base = [
      ['context','Execution context','context'],['plan','Agent plan','agent'],['engine','Deterministic engine','tool'],['branch-a','Primary diagnostics','test'],['branch-b','Robustness branch','test'],['evidence','Evidence bundle','evidence'],['review','Agent review','agent'],['governance','Governance','governance'],['attest','Sign-off','attestation']
    ] as const
    state.graph.nodes=base.map(([id,label,kind],i)=>({id,runId:state.snapshot.runId,label,kind,status:(i===0?'completed':'future'),parentId:i?base[i-1][0]:undefined,subtitle:i===0?request.contextId:undefined}))
    state.graph.edges=base.slice(1).map((n,i)=>({id:`edge-${i}`,source:base[i][0],target:n[0],relation:(n[0].startsWith('branch')?'branch':'next') as any}))
    if(request.parentRunId){ state.graph.nodes.unshift({id:'parent-run',runId:state.snapshot.runId,label:`Parent ${request.parentRunId}`,kind:'human',status:'completed',subtitle:'Iteration lineage'}); state.graph.edges.unshift({id:'edge-parent',source:'parent-run',target:'context',relation:'rerun'}) }
  }
  private scheduleRun(state:DemoRunState, request:RunRequest){
    const steps=[
      {ms:350,node:'plan',phase:'planning',title:'Plan accepted',message:'Agent translated the goal into an explicit deterministic execution plan.',p:8},
      {ms:850,node:'engine',phase:'running',title:'Deterministic engine started',message:'Execution context validated. Analytical work is now running.',p:18},
      {ms:1450,node:'branch-a',phase:'running',title:'Primary diagnostics',message:'Running compatible deterministic test surfaces and recording evidence.',p:36},
      {ms:2150,node:'branch-b',phase:'running',title:'Robustness branch opened',message:'A separate stress branch is executing because it is compatible with this workflow.',p:58},
      {ms:3000,node:'evidence',phase:'partial',title:'Evidence bundle created',message:'Quantitative outputs have been written as traceable EvidenceRecords.',p:74},
      {ms:3650,node:'review',phase:'waiting_ai',title:'Agent evidence review',message:'Qualitative review is reading Evidence IDs; deterministic numbers remain authoritative.',p:86},
      {ms:4350,node:'governance',phase:'running',title:'Governance evaluation',message:'Policy and governance layers are evaluating grounded evidence.',p:94},
      {ms:5050,node:'attest',phase:'completed',title:'Run signed off',message:'Attestation created. The complete parent path remains inspectable.',p:100},
    ]
    steps.forEach((s,idx)=>{
      const t=window.setTimeout(()=>{
        state.snapshot.phase=s.phase as any; state.snapshot.statusLabel=s.title; state.snapshot.elapsedMs=s.ms
        state.snapshot.progress={label:s.title,percent:s.p,completed:idx+1,total:steps.length,detail:s.message}
        const node=state.graph.nodes.find(n=>n.id===s.node); if(node) node.status='running'
        state.graph.nodes.forEach(n=>{ if(n.id!==s.node && n.status==='running') n.status='completed' })
        this.emit(state,{type:'phase',nodeId:s.node,title:s.title,message:s.message,status:s.phase==='completed'?'completed':'running',progress:state.snapshot.progress})
        if(s.node==='engine') this.emit(state,{type:'tool_started',nodeId:'engine',parentNodeId:'plan',title:'Tool · canonical test coordinator',message:'Resolved compatible registered tests from the backend capability map.',status:'running',metadata:{tool:'registered_test_coordinator'}})
        if(s.node==='branch-a') this.addEvidence(state,'EV-DATA','data.integrity.schema','Data contract integrity','PASS',[{name:'rows_evaluated',value:12000},{name:'schema_violations',value:0}], 'branch-a')
        if(s.node==='branch-b') this.addEvidence(state,'EV-ROB','robustness.perturbation','Perturbation stability','ATTENTION',[{name:'baseline_score',value:0.842},{name:'stressed_score',value:0.796}], 'branch-b')
        if(s.node==='evidence') this.addEvidence(state,'EV-CAL','calibration.reliability','Calibration reliability','RECORDED',[{name:'brier_score',value:0.118}], 'branch-a')
        if(s.node==='evidence') state.artifacts.push({artifactId:'ART-CURVE',runId:state.snapshot.runId,label:'Reliability detail',kind:'table',mimeType:'application/json',createdAt:now(),description:'Generated from the deterministic calibration surface.',preview:{type:'key-value',payload:{bins:10,source:'EV-CAL'}}})
        if(s.node==='review') state.findings=[{findingId:'F-ROB',runId:state.snapshot.runId,title:'Stress branch deserves inspection',summary:'The robustness branch produced an attention-status EvidenceRecord. No business threshold is invented; inspect the evidence and choose a follow-up if useful.',evidenceIds:['EV-ROB'],sourceNodeId:'branch-b',severity:'attention',limitations:['Preview adapter uses deterministic synthetic context.'],availableActions:['explain','challenge','deeper_test','rerun']}]
        if(s.node==='governance') state.governance={disposition:'ACCEPT_WITH_CONDITIONS',policyDecision:'ALLOW',rationale:'Grounded evidence available; one attention item remains explicitly visible.',evidenceCoverage:1,unresolvedItems:['Inspect robustness evidence before operational adoption.']}
        if(s.node==='attest'){ if(node) node.status='completed'; state.attestation={merkleRoot:'preview:7fd2a8…',createdAt:now(),evidenceCount:state.evidence.length,artifactCount:state.artifacts.length,reproducibilityId:'PREVIEW-DET-42'} }
      },s.ms); state.timers.push(t)
    })
  }
  private addEvidence(state:DemoRunState,evidenceId:string,testId:string,title:string,status:EvidenceRecord['status'],metrics:EvidenceRecord['metrics'],parentNodeId:string){
    const ev={evidenceId,runId:state.snapshot.runId,testId,title,status,metrics,provenance:[`run:${state.snapshot.runId}`,`test:${testId}`],parentNodeId,createdAt:now(),summary:`Deterministic preview evidence from ${testId}.`} as EvidenceRecord
    state.evidence.push(ev); const node=state.graph.nodes.find(n=>n.id===parentNodeId); if(node) node.evidenceIds=[...(node.evidenceIds||[]),evidenceId]
    this.emit(state,{type:'evidence_created',nodeId:parentNodeId,title:`Evidence · ${title}`,message:`${evidenceId} created by ${testId}.`,status:'completed',evidenceIds:[evidenceId]})
  }
}
