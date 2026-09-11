import type { StartBackend, StreamSubscription } from '../../contracts/backend'
import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, CheckpointRecord,
  DecisionReceipt, EdaProfile, EvidenceRecord, ExecutionContext, ExecutionGraph,
  Finding, GovernanceState, ProposedAction, QuestionResponse,
  RunCompareResult, RunHistoryItem, RunLineage,
  RunRequest, RunSnapshot, RuntimeEvent, ScenarioItem, StARTCapabilityManifest, TestCatalogItem, WorkflowId
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
  ['recommender_system','Recommender Systems','Evaluate collaborative filtering, ranking, and factorization models.','ml'],
].map(([id,label,description,category]) => ({id:id as WorkflowId,label,description,category:category as 'ml'|'quant',enabled:id!=='model_comparison',disabledReason:id==='model_comparison'?'Multi-model candidate comparison workflow requires multi-candidate input protocol not yet enabled in the canonical web review interface.':undefined}))

const contexts: ExecutionContext[] = [
  { id:'institutional_credit_v1', label:'Synthetic Binary Classification Benchmark', kind:'dataset', description:'Seeded tabular binary classification benchmark for predictive risk model validation.', provenance:'Built-in deterministic synthetic generator', shape:'500 × 8', target:'target', seed:42, badges:['public-safe','seeded','binary','benchmark'] },
  { id:'deep_learning_v1', label:'Synthetic Tabular Neural Latent Benchmark', kind:'dataset', description:'Seeded tabular neural network benchmark for deep learning performance, sensitivity, and calibration diagnostics.', provenance:'Built-in deterministic synthetic generator', shape:'500 × 8', target:'target', seed:17, badges:['public-safe','deep-learning','tabular','neural'] },
  { id:'institutional_market_v1', label:'Synthetic Multi-Asset Market World', kind:'synthetic-world', description:'Seeded multi-asset scenario context for traded risk, VaR backtests, and portfolio optimization workflows.', provenance:'Built-in deterministic synthetic market generator', shape:'50 assets × 1,000 observations', target:'N/A', seed:7, badges:['public-safe','quantitative','var-backtest','portfolio'] },
]

const mkPlan = (workflow:WorkflowId) => {
  const map: Record<WorkflowId, string[]> = {
    fraud_anomaly_aml: [],
    predictive_ml:['Context prepared','Preflight checks','Feature diagnostics','Supervised evaluation','Explainability','Evidence assembly','Governance','Attestation'],
    deep_learning:['Context prepared','Inspect architecture','Initialize training diagnostics','Observe epoch/batch path','Run robustness branch','Generate interpretability evidence','Evidence assembly','Governance & sign-off'],
    data_diagnostics:['Context prepared','Validate schema','Inspect missingness','Inspect drift & distribution','Inspect feature structure','Create evidence bundle','Evidence review','Sign-off'],
    model_diagnostics:['Context prepared','Resolve model context','Inspect error structure','Inspect residual behaviour','Inspect stability','Create evidence bundle','Evidence review','Sign-off'],
    calibration:['Context prepared','Resolve score semantics','Measure calibration','Inspect reliability structure','Compare calibration states','Create evidence bundle','Review & sign-off'],
    robustness:['Context prepared','Resolve perturbation plan','Execute stress cases','Compare degradation paths','Create evidence bundle','Review & sign-off'],
    explainability:['Context prepared','Resolve compatible explainers','Generate attribution evidence','Inspect local/global structure','Create artifacts','Review & sign-off'],
    hyperparameter_tuning:['Context prepared','Validate search space','Establish baseline','Execute bounded trials','Compare candidates','Create evidence bundle','Review & sign-off'],
    model_comparison:['Context prepared','Resolve candidate set','Establish shared protocol','Evaluate candidates','Compare evidence','Create decision bundle','Review & sign-off'],
    quantitative_finance:['Load market world','Validate portfolio context','Build analytical plan','Run scenario & stress branches','Run portfolio/risk checks','Create evidence bundle','Review & governance','Attestation'],
    recommender_system:['Context prepared','Prepare interaction matrix','Train recommender algorithm','Evaluate ranking metrics (NDCG@K)','Evaluate coverage & cold-start','Sensitivity analysis','Evidence assembly','Governance & sign-off'],
  }
  return map[workflow].map((label,i)=>({id:`step-${i+1}`,label,kind:(i===0?'context':i===1?'test':i===6?'governance':i===7?'attestation':'test') as any,status:'queued' as any,parentId:i?`step-${i}`:undefined}))
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
  async getCapabilityManifest(): Promise<StARTCapabilityManifest> {
    return {
      workbench: {
        name: 'StART — Agentic AI Engineering Workbench',
        tagline: 'Build · Tune · Stress · Explain · Compare · Govern',
        version: '4.0.0',
        default_execution_mode: 'hybrid_workbench',
        deterministic_science_invariant: true,
      },
      execution_modes: [
        { id: 'hybrid_workbench', label: 'Hybrid Workbench', is_default: true, description: 'Generative AI plan synthesis and reasoning combined with 100% deterministic science execution engines.' },
        { id: 'agentic_session', label: 'Agentic Session', is_default: false, description: 'Deliberative multi-agent committee exploration (12 specialized engineering agents).' },
        { id: 'deterministic_run', label: 'Deterministic Run', is_default: false, description: 'Direct, reproducible parameterized pipeline execution with offline determinism.' },
      ],
      domains: {},
      deferred_capabilities: [],
      ai_provider: { provider: 'openai', model: 'gpt-5.1', status: 'ONLINE', source: 'demo', strict_zero_substitution: true },
    }
  }
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
  async listScenarios(): Promise<ScenarioItem[]> {
    return [
      {
        id: 'institutional_credit_v1',
        label: 'Synthetic Binary Classification Benchmark',
        domain: 'predictive_ml',
        classification: 'CANONICAL_EXECUTION_CONTEXT',
        compatible_context_id: 'institutional_credit_v1',
        compatible_workflows: ['predictive_ml'],
        shape: '500 × 8',
        rows: 500,
        features: 8,
        target: 'target',
        categories: ['Calibration', 'Robustness', 'Explainability'],
        registered_tests_count: 52,
        applicable_tests_count: 52,
        generator_identity: 'start.runtime.contexts.instantiate_context',
        seed: 42,
        description: 'Seeded tabular binary classification benchmark.',
        provenance_note: 'Canonical context seed 42',
      },
    ]
  }
  async getScenarioEda(scenarioId: string): Promise<EdaProfile> {
    return {
      domain: 'tabular',
      provenance: {
        scenario_id: scenarioId,
        context_id: 'institutional_credit_v1',
        classification: 'CANONICAL_EXECUTION_CONTEXT',
        generator_identity: 'demo.fixture',
        seed: 42,
        input_shape: '500 × 8',
        profiling_operation: 'deterministic_descriptive_profiling',
        sha256_fingerprint: 'SHA256:demo_fixture_only',
        provenance_note: 'Demo offline fixture',
      },
      overview: { rows: 500, columns: 9, feature_count: 8, target: 'target', missing_cells: 0, duplicate_rows: 0, memory_kb: 40 },
    }
  }
  async getContextEda(contextId: string): Promise<EdaProfile> {
    return this.getScenarioEda(contextId)
  }
  async listTests(): Promise<TestCatalogItem[]> {
    return []
  }
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
  async getCheckpoints(runId:string): Promise<CheckpointRecord[]> {
    return [
      { checkpoint_id: 'CP-001', name: 'Context ready', status: 'completed', producing_stage: 'step-1', agent_signature: 'Director', evidence_ids: [], commit_hash: 'c01a98', timestamp: now(), summary: 'Execution context prepared.' },
      { checkpoint_id: 'CP-002', name: 'Preflight complete', status: 'completed', producing_stage: 'step-2', agent_signature: 'DataSpecialist', evidence_ids: ['EV-DATA'], commit_hash: 'c02b87', timestamp: now(), summary: 'Data contract verified.' },
      { checkpoint_id: 'CP-003', name: 'Feature analysis', status: 'completed', producing_stage: 'step-3', agent_signature: 'FeatureSpecialist', evidence_ids: ['EV-FEAT'], commit_hash: 'c03c76', timestamp: now(), summary: 'Distribution drift checked.' },
      { checkpoint_id: 'CP-004', name: 'Model evaluation', status: 'completed', producing_stage: 'step-4', agent_signature: 'ModelSpecialist', evidence_ids: ['EV-PERF', 'EV-CAL'], commit_hash: 'c04d65', timestamp: now(), summary: 'Model performance measured.' },
      { checkpoint_id: 'CP-005', name: 'Evidence committed', status: 'completed', producing_stage: 'step-6', agent_signature: 'EvidenceLedger', evidence_ids: ['EV-DATA', 'EV-FEAT', 'EV-PERF', 'EV-CAL', 'EV-ROB'], commit_hash: 'c05e54', timestamp: now(), summary: 'Evidence surfaces sealed.' },
      { checkpoint_id: 'CP-006', name: 'Attestation signed', status: 'completed', producing_stage: 'step-8', agent_signature: 'ModelGovernance', evidence_ids: ['EV-ROB'], commit_hash: '0x7fd2a8...', timestamp: now(), summary: 'Attestation seal signed.' },
    ]
  }
  async listRuns(query?: { workflow?: string; status?: string; domain?: string }): Promise<RunHistoryItem[]> {
    const list: RunHistoryItem[] = []
    for (const [id, state] of this.runs.entries()) {
      if (query?.workflow && state.snapshot.workflowId !== query.workflow) continue
      if (query?.status && state.snapshot.phase !== query.status) continue
      list.push({
        run_id: id,
        workflow: state.snapshot.workflowId,
        domain: 'predictive_ml',
        context_id: state.snapshot.contextId,
        created_at: Date.parse(state.snapshot.startedAt),
        status: state.snapshot.phase as any,
        evidence_count: state.evidence.length,
        artifact_count: state.artifacts.length,
        governance_disposition: state.governance?.disposition || '',
        parent_run_id: state.snapshot.parentRunId,
        intervention: (state.snapshot as any).intervention || null,
        goal: state.snapshot.goal,
      })
    }
    return list
  }
  async compareRuns(runA: string, runB: string): Promise<RunCompareResult> {
    const a = this.must(runA)
    const b = this.must(runB)
    return {
      compatible: true,
      runA: {
        runId: runA,
        workflow: a.snapshot.workflowId,
        domain: 'predictive_ml',
        contextId: a.snapshot.contextId,
        status: a.snapshot.phase,
        evidenceCount: a.evidence.length,
        artifactCount: a.artifacts.length,
        governanceDisposition: a.governance?.disposition || '',
      },
      runB: {
        runId: runB,
        workflow: b.snapshot.workflowId,
        domain: 'predictive_ml',
        contextId: b.snapshot.contextId,
        status: b.snapshot.phase,
        evidenceCount: b.evidence.length,
        artifactCount: b.artifacts.length,
        governanceDisposition: b.governance?.disposition || '',
      },
      metricComparisons: [],
      findings: [],
      artifacts: [],
    }
  }
  async getRunLineage(runId: string): Promise<RunLineage> {
    const target = this.must(runId)
    const children: any[] = []
    for (const [id, state] of this.runs.entries()) {
      if (state.snapshot.parentRunId === runId) {
        children.push({
          runId: id,
          createdAt: Date.parse(state.snapshot.startedAt),
          status: state.snapshot.phase,
          intervention: (state.snapshot as any).intervention || null,
        })
      }
    }
    return {
      runId,
      parentRunId: target.snapshot.parentRunId,
      intervention: (target.snapshot as any).intervention || null,
      children,
    }
  }
  async searchGlobal(query: string): Promise<Array<{ id: string; category: string; title: string; subtitle?: string; data?: any }>> {
    const q = query.toLowerCase().trim()
    const results: Array<{ id: string; category: string; title: string; subtitle?: string; data?: any }> = []
    for (const [id, state] of this.runs.entries()) {
      if (id.toLowerCase().includes(q) || state.snapshot.workflowId.toLowerCase().includes(q)) {
        results.push({
          category: 'run',
          id,
          title: `Run ${id}`,
          subtitle: `${state.snapshot.workflowId} (${state.snapshot.contextId})`,
          data: state.snapshot,
        })
      }
    }
    return results
  }
  async recordDecision(runId:string, decision: Partial<DecisionReceipt>): Promise<DecisionReceipt> {

    return {
      receipt_id: id('REC'),
      run_id: runId,
      action: (decision.action || 'ACCEPT') as any,
      target_stage: decision.target_stage || 'step-4',
      evidence_ids: decision.evidence_ids || ['EV-PERF'],
      rationale: decision.rationale || 'Human decision recorded.',
      author: 'Risk Officer',
      decision_hash: id('HASH'),
      timestamp: now(),
      status: 'RECORDED',
      immutable_evidence_preserved: true,
    }
  }
  async askQuestion(runId:string, query: { question: string; targetStage?: string; evidenceId?: string }): Promise<QuestionResponse> {
    return {
      question: query.question,
      answer: `Based on evidence analysis: EV-PERF shows ROC-AUC=0.892 (PASS). All statistical assertions conform to protocol specifications.`,
      citations: [{ evidence_id: 'EV-PERF', test_id: 'supervised.evaluation.roc', status: 'PASS', metrics: { roc_auc: 0.892 }, snippet: 'ROC-AUC=0.892 PASS' }],
      targetStage: query.targetStage,
      targetEvidenceId: query.evidenceId,
      receipt: {
        receipt_id: id('REC-Q'),
        run_id: runId,
        action: 'QUESTION',
        evidence_ids: ['EV-PERF'],
        rationale: query.question,
        author: 'Risk Officer',
        decision_hash: id('HASH'),
        timestamp: now(),
        status: 'ANSWERED',
        immutable_evidence_preserved: true,
      },
      timestamp: now(),
    }
  }
  private must(runId:string){ const s=this.runs.get(runId); if(!s) throw new Error(`Unknown run ${runId}`); return s }
  private emit(state:DemoRunState,event:Omit<RuntimeEvent,'eventId'|'sequence'|'timestamp'|'runId'>){
    const e:RuntimeEvent={...event,eventId:id('EVT'),sequence:state.events.length+1,timestamp:now(),runId:state.snapshot.runId}
    state.events.push(e); state.snapshot.updatedAt=e.timestamp; state.snapshot.progress=e.progress||state.snapshot.progress
    state.listeners.forEach(l=>l(structuredClone(e)))
  }
  private seedGraph(state:DemoRunState, request:RunRequest){
    const base = [
      ['step-1','Context prepared','context'],
      ['step-2','Preflight checks','test'],
      ['step-3','Feature diagnostics','test'],
      ['step-4','Supervised evaluation','test'],
      ['step-5','Explainability','test'],
      ['step-6','Evidence assembly','evidence'],
      ['step-7','Governance','governance'],
      ['step-8','Attestation','attestation']
    ] as const
    state.graph.nodes=base.map(([id,label,kind],i)=>({id,runId:state.snapshot.runId,label,kind,status:'future' as any,parentId:i?base[i-1][0]:undefined,subtitle:i===0?request.contextId:undefined}))
    state.graph.edges=base.slice(1).map((n,i)=>({id:`edge-${i}`,source:base[i][0],target:n[0],relation:'next' as any}))
    if(request.parentRunId){ state.graph.nodes.unshift({id:'parent-run',runId:state.snapshot.runId,label:`Parent ${request.parentRunId}`,kind:'human',status:'completed',subtitle:'Iteration lineage'}); state.graph.edges.unshift({id:'edge-parent',source:'parent-run',target:'step-1',relation:'rerun'}) }
  }
  private scheduleRun(state:DemoRunState, request:RunRequest){
    const steps=[
      {ms:300,node:'step-1',phase:'running',title:'Context prepared',message:'Deterministic execution context loaded and verified. 500 samples × 8 features.',p:12},
      {ms:750,node:'step-2',phase:'running',title:'Preflight checks',message:'Data contract, column typing, and schema integrity verified across deterministic boundaries.',p:25},
      {ms:1350,node:'step-3',phase:'running',title:'Feature diagnostics',message:'Feature collinearity, distribution drift, and missingness evaluated.',p:40},
      {ms:2050,node:'step-4',phase:'running',title:'Supervised evaluation',message:'Deterministic classification metrics and probability calibration measured.',p:58},
      {ms:2850,node:'step-5',phase:'running',title:'Explainability & robustness',message:'Perturbation stress testing and SHAP attribution tensors computed.',p:75},
      {ms:3550,node:'step-6',phase:'partial',title:'Evidence assembly',message:'Deterministic EvidenceRecords and analytical artifacts sealed into run bundle.',p:88},
      {ms:4200,node:'step-7',phase:'running',title:'Governance evaluation',message:'Model risk policy rules checked against grounded evidence records.',p:95},
      {ms:4900,node:'step-8',phase:'completed',title:'Attestation sealed',message:'Merkle tree attestation root computed and signed. All parent paths preserved.',p:100},
    ]
    steps.forEach((s,idx)=>{
      const t = setTimeout(()=>{
        state.snapshot.phase=s.phase as any; state.snapshot.statusLabel=s.title; state.snapshot.elapsedMs=s.ms
        state.snapshot.progress={label:s.title,percent:s.p,completed:idx+1,total:steps.length,detail:s.message}
        const node=state.graph.nodes.find(n=>n.id===s.node); if(node) node.status='running'
        state.graph.nodes.forEach(n=>{ if(n.id!==s.node && n.status==='running') n.status='completed' })
        this.emit(state,{type:'phase',nodeId:s.node,title:s.title,message:s.message,status:s.phase==='completed'?'completed':'running',progress:state.snapshot.progress})

        if(s.node==='step-2') {
          this.emit(state,{type:'tool_started',nodeId:'step-2',parentNodeId:'step-1',title:'Tool · schema validator',message:'Checked 8 feature columns against canonical schema contract.',status:'completed',metadata:{tool:'schema_validator'}})
          this.addEvidence(state,'EV-DATA','data.integrity.schema','Data contract integrity','PASS',[{name:'rows_evaluated',value:500},{name:'schema_violations',value:0}], 'step-2')
        }
        if(s.node==='step-3') {
          this.addEvidence(state,'EV-FEAT','feature.missingness.drift','Feature distribution integrity','PASS',[{name:'missing_rate',value:0.0},{name:'feature_count',value:8}], 'step-3')
        }
        if(s.node==='step-4') {
          this.addEvidence(state,'EV-PERF','supervised.evaluation.roc','ROC-AUC discrimination','PASS',[{name:'roc_auc',value:0.892},{name:'accuracy',value:0.846},{name:'f1_score',value:0.838}], 'step-4')
          this.addEvidence(state,'EV-CAL','calibration.reliability','Brier score reliability','PASS',[{name:'brier_score',value:0.118},{name:'expected_calibration_error',value:0.042}], 'step-4')
        }
        if(s.node==='step-5') {
          this.addEvidence(state,'EV-ROB','robustness.perturbation','Perturbation stability stress test','ATTENTION',[{name:'baseline_score',value:0.892},{name:'stressed_score',value:0.814},{name:'degradation',value:0.078}], 'step-5')
          state.artifacts.push({artifactId:'ART-SHAP',runId:state.snapshot.runId,label:'SHAP feature attributions',kind:'table',mimeType:'application/json',createdAt:now(),description:'Computed via deterministic tree SHAP explainer.',preview:{type:'key-value',payload:{top_feature:'feature_0',attribution:0.342,mean_abs_shap:0.188}}})
          this.emit(state,{type:'artifact_created',nodeId:'step-5',title:'Artifact · SHAP feature attributions',message:'ART-SHAP generated from deterministic explainability engine.',status:'completed',artifactIds:['ART-SHAP']})
        }
        if(s.node==='step-6') {
          state.artifacts.push({artifactId:'ART-CURVE',runId:state.snapshot.runId,label:'Reliability curve bins',kind:'table',mimeType:'application/json',createdAt:now(),description:'Binned calibration probabilities across 10 deciles.',preview:{type:'key-value',payload:{bins:10,mean_pred:0.504,mean_true:0.496}}})
          this.emit(state,{type:'artifact_created',nodeId:'step-6',title:'Artifact · Reliability curve bins',message:'ART-CURVE generated from calibration engine.',status:'completed',artifactIds:['ART-CURVE']})
          state.findings=[{
            findingId:'F-ROB',
            runId:state.snapshot.runId,
            title:'Stress perturbation threshold flagged for review',
            summary:'The robustness branch produced an attention-status EvidenceRecord (EV-ROB). Under Gaussian feature perturbation, ROC-AUC degraded by 7.8% (0.892 → 0.814). No arbitrary threshold was invented; inspect the evidence and choose a follow-up action if desired.',
            evidenceIds:['EV-ROB'],
            sourceNodeId:'step-5',
            severity:'attention',
            limitations:['Deterministic benchmark synthetic context with 500 samples.'],
            availableActions:['explain','challenge','deeper_test','rerun']
          }]
          this.emit(state,{type:'finding_created',nodeId:'step-6',title:'Finding · Robustness attention',message:'Finding F-ROB grounded in EV-ROB recorded.',status:'completed'})
        }
        if(s.node==='step-7') {
          state.governance={
            disposition:'ACCEPT_WITH_CONDITIONS',
            policyDecision:'ALLOW',
            rationale:'All preflight and performance checks passed. Perturbation degradation of 7.8% noted in F-ROB; accepted with requirement for operational monitoring.',
            evidenceCoverage:1.0,
            unresolvedItems:['Inspect robustness evidence EV-ROB before high-volume deployment.']
          }
          this.emit(state,{type:'governance',nodeId:'step-7',title:'Governance · Policy evaluation',message:'Disposition ACCEPT_WITH_CONDITIONS recorded.',status:'completed'})
        }
        if(s.node==='step-8') {
          if(node) node.status='completed'
          state.snapshot.phase='completed'
          state.snapshot.statusLabel='Run signed off'
          state.attestation={
            merkleRoot:'0x7fd2a8e391b490fc6a414e21a0899c75902bc5b4d762e847c2da2834bce96841',
            createdAt:now(),
            evidenceCount:state.evidence.length,
            artifactCount:state.artifacts.length,
            reproducibilityId:'START-DET-42'
          }
          this.emit(state,{type:'attested',nodeId:'step-8',title:'Attestation · Merkle seal',message:'Merkle tree signed and sealed. Run finalized.',status:'completed'})
          this.emit(state,{type:'run_completed',nodeId:'step-8',title:'Execution finished',message:'Deterministic run signed off with complete parent lineage.',status:'completed',progress:state.snapshot.progress})
        }
      },s.ms); state.timers.push(t)
    })
  }
  private addEvidence(state:DemoRunState,evidenceId:string,testId:string,title:string,status:EvidenceRecord['status'],metrics:EvidenceRecord['metrics'],parentNodeId:string){
    const ev={evidenceId,runId:state.snapshot.runId,testId,title,status,metrics,provenance:[`run:${state.snapshot.runId}`,`test:${testId}`,`node:${parentNodeId}`],parentNodeId,createdAt:now(),summary:`Deterministic verification evidence from ${testId}.`} as EvidenceRecord
    state.evidence.push(ev); const node=state.graph.nodes.find(n=>n.id===parentNodeId); if(node) node.evidenceIds=[...(node.evidenceIds||[]),evidenceId]
    this.emit(state,{type:'evidence_created',nodeId:parentNodeId,title:`Evidence · ${title}`,message:`${evidenceId} created by ${testId}.`,status:'completed',evidenceIds:[evidenceId]})
  }
}
