import { useState } from 'react'
import { ArrowRight, CheckCircle, Database, Globe, FileText } from 'lucide-react'
import type { AgentPlanPreview, Capability, ExecutionContext, ExecutionMode, ScenarioItem, StARTCapabilityManifest, WorkflowId } from '../../contracts/types'
import { DataValue } from '../investigation/Science'
import { EngineeringControls, algorithmContexts } from '../capabilities/EngineeringControls'
import { compatibleContexts } from '../capabilities/runtimeSupport'
import { liveRequest } from '../certification/liveApi'

export function Composer(p: {
  capabilities: Capability[];
  contexts: ExecutionContext[];
  scenarios: ScenarioItem[];
  selectedScenario: ScenarioItem | null;
  onSelectScenario: (s: ScenarioItem) => void;
  workflow: WorkflowId | null;
  setWorkflow: (v: WorkflowId) => void;
  context: string | null;
  setContext: (v: string) => void;
  goal: string;
  setGoal: (v: string) => void;
  plan: AgentPlanPreview | null;
  onPlan: () => void;
  onStart: () => void;
  busy: boolean;
  adapterMode: string;
  onOpenTestCatalog: () => void;
  executionMode?: ExecutionMode;
  onExecutionModeChange?: (m: ExecutionMode) => void;
  profileMode?: 'workbench' | 'enterprise';
  manifest?: StARTCapabilityManifest | null;
  parameters?: Record<string, any>;
  onParametersChange?: (p: Record<string, any>) => void;
  seed?: number;
  onSeedChange?: (v: number | undefined) => void;
  trials?: number;
  onTrialsChange?: (v: number | undefined) => void;
}) {
  const [query, setQuery] = useState('')
  const [hubTab, setHubTab] = useState<'builtin' | 'live' | 'local'>('builtin')

  // Live provider controls state
  const [provider, setProvider] = useState<'huggingface' | 'openml' | 'uci' | 'kaggle'>('huggingface')
  const [datasetId, setDatasetId] = useState('scikit-learn/adult-census-income')
  const [revision, setRevision] = useState('')
  const [target, setTarget] = useState('')
  const [contract, setContract] = useState<any>(null)
  const [report, setReport] = useState<any>(null)
  const [liveBusy, setLiveBusy] = useState<string | null>(null)
  const [liveError, setLiveError] = useState<string | null>(null)

  // Local file controls state
  const [localPath, setLocalPath] = useState('data/credit_risk.csv')

  const contexts = compatibleContexts(p.contexts, p.scenarios, p.workflow)

  const isLive = p.context && !contexts.some(c => c.id === p.context)
  const dynamicLiveContext = isLive && p.context ? {
    id: p.context,
    label: p.context,
    description: `Live provider execution dataset: ${p.context}`,
    provenance: `Live Provider Ingestion (${p.parameters?.provider || provider})`,
    shape: contract ? `${contract.rows_declared ?? 'dynamic'} rows × ${contract.features?.length ?? 'dynamic'} cols` : 'Dynamic remote stream',
    target: target || contract?.target_column || 'Normalized target (0/1)',
    seed: p.seed ?? 42,
  } : null
  const context = contexts.find(c => c.id === p.context) || dynamicLiveContext

  const inspectedScenario = p.selectedScenario && p.selectedScenario.id !== p.context
  const mismatchedAlgorithm = p.workflow === 'recommender_system' && (!p.parameters?.algorithm || algorithmContexts[p.parameters.algorithm] !== p.context)
  const ready = !mismatchedAlgorithm && !!p.workflow && !!context && !!p.goal.trim() && !p.busy && !inspectedScenario
  const planCurrent = p.plan && p.plan.goal === p.goal && p.plan.workflowId === p.workflow && p.plan.contextId === p.context

  const setProviderAndPreset = (newProv: 'huggingface' | 'openml' | 'uci' | 'kaggle') => {
    setProvider(newProv)
    if (newProv === 'huggingface') setDatasetId('scikit-learn/adult-census-income')
    else if (newProv === 'openml') setDatasetId('credit-g')
    else if (newProv === 'uci') setDatasetId('iris')
    else if (newProv === 'kaggle') setDatasetId('titanic')
    setContract(null)
    setReport(null)
    setLiveError(null)
  }

  const handleResolveLive = async () => {
    if (!datasetId.trim()) return
    setLiveBusy('Resolving remote dataset contract...')
    setLiveError(null)
    try {
      const data = await liveRequest('/data/resolve', {
        method: 'POST',
        body: {
          provider,
          dataset_id: datasetId.trim(),
          revision: revision.trim() || undefined,
          target: target.trim() || undefined,
        },
      })
      setContract(data.contract)
      setReport(null)
    } catch (err: any) {
      setLiveError(err.message || 'Failed to resolve dataset contract.')
    } finally {
      setLiveBusy(null)
    }
  }

  const handlePrecertifyLive = async () => {
    if (!datasetId.trim()) return
    setLiveBusy('Evaluating pre-certification checks...')
    setLiveError(null)
    try {
      const data = await liveRequest('/data/precertify', {
        method: 'POST',
        body: {
          provider,
          dataset_id: datasetId.trim(),
          revision: revision.trim() || undefined,
          target: target.trim() || undefined,
          sample_rows: 50,
        },
      })
      setReport(data)
    } catch (err: any) {
      setLiveError(err.message || 'Pre-certification evaluation failed.')
    } finally {
      setLiveBusy(null)
    }
  }

  const handleSelectLiveForExecution = () => {
    const chosenId = datasetId.trim()
    p.setContext(chosenId)
    p.onParametersChange?.({
      ...p.parameters,
      dataset_id: chosenId,
      dataset: chosenId,
      provider,
    })
  }

  const handleSelectLocalForExecution = () => {
    const chosenPath = localPath.trim()
    p.setContext(chosenPath)
    p.onParametersChange?.({
      ...p.parameters,
      dataset_id: chosenPath,
      dataset: chosenPath,
      is_local: true,
    })
  }

  return (
    <div className="define-workspace">
      <div className="eyebrow">NEW WORKSPACE / DEFINE</div>
      <h1>What are you engineering?</h1>
      <p className="define-intro">Build · Tune · Stress · Explain · Compare · Govern</p>

      <div className="define-form">
        {/* 01 Engineering objective */}
        <div className="define-row">
          <span className="field-number">01</span>
          <label className="stacked-label">
            Engineering objective
            <textarea
              value={p.goal}
              onChange={e => p.setGoal(e.target.value)}
              placeholder="Describe the question, experiment, and constraints."
            />
          </label>
        </div>

        {/* 02 Execution mode */}
        <div className="define-row">
          <span className="field-number">02</span>
          <div>
            <label>Execution mode</label>
            <div className="execution-mode-choices">
              {p.manifest?.execution_modes.map(m => (
                <button
                  key={m.id}
                  className={p.executionMode === m.id ? 'primary' : 'tonal'}
                  aria-pressed={p.executionMode === m.id}
                  onClick={() => p.onExecutionModeChange?.(m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <p className="field-help">
              {p.executionMode === 'agentic_session'
                ? 'Backend plan proposal and executive synthesis checkpoint, followed by deterministic execution.'
                : p.executionMode === 'deterministic_run'
                ? 'Deterministic execution. Zero automatic AI calls.'
                : 'Inspect the backend proposal, edit configuration, then execute deterministic tools. Evidence-grounded AI questions remain optional.'}
            </p>
          </div>
        </div>

        {/* 03 Domain / executable workflow */}
        <div className="define-row">
          <span className="field-number">03</span>
          <label className="stacked-label">
            Domain / executable workflow
            <select
              value={p.workflow ?? ''}
              onChange={e => p.setWorkflow(e.target.value as WorkflowId)}
            >
              <option value="" disabled>Select a workflow</option>
              {p.capabilities.filter(c => c.enabled).map(c => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </label>
        </div>

        {/* 04 Dataset Hub */}
        <div className="define-row">
          <span className="field-number">04</span>
          <div>
            <label className="stacked-label">Dataset Hub</label>
            <div className="execution-mode-choices" role="tablist" aria-label="Dataset Hub Sources" style={{ marginBottom: '10px' }}>
              <button
                role="tab"
                type="button"
                className={hubTab === 'builtin' ? 'primary' : 'tonal'}
                aria-selected={hubTab === 'builtin'}
                onClick={() => setHubTab('builtin')}
              >
                <Database size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
                Built-in
              </button>
              <button
                role="tab"
                type="button"
                className={hubTab === 'live' ? 'primary' : 'tonal'}
                aria-selected={hubTab === 'live'}
                onClick={() => setHubTab('live')}
              >
                <Globe size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
                Live Providers
              </button>
              <button
                role="tab"
                type="button"
                className={hubTab === 'local' ? 'primary' : 'tonal'}
                aria-selected={hubTab === 'local'}
                onClick={() => setHubTab('local')}
              >
                <FileText size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
                Local File
              </button>
            </div>

            {/* Tab 1: Built-in */}
            {hubTab === 'builtin' && (
              <div>
                <label className="stacked-label">
                  Canonical execution dataset
                  <select
                    value={contexts.some(c => c.id === p.context) ? (p.context ?? '') : ''}
                    onChange={e => {
                      p.setContext(e.target.value)
                      p.onParametersChange?.({ ...p.parameters, dataset_id: e.target.value })
                    }}
                  >
                    <option value="" disabled>Select a compatible execution dataset</option>
                    {contexts.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select>
                </label>
                {p.workflow && !contexts.length && <p role="status">No compatible execution dataset supplied.</p>}
                {context && !isLive && (
                  <>
                    <p className="field-help">{context.description}</p>
                    <DataValue value={{ source: context.provenance, shape: context.shape, target: context.target, seed: context.seed }} />
                  </>
                )}
                <details style={{ marginTop: '8px' }}>
                  <summary>Inspect dataset and scenario catalog</summary>
                  <input
                    type="search"
                    aria-label="Search dataset catalog"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="Name, source, task or target"
                  />
                  {p.scenarios
                    .filter(s => (!p.workflow || s.compatible_workflows.includes(p.workflow)) && JSON.stringify(s).toLowerCase().includes(query.toLowerCase()))
                    .map(s => (
                      <article className="catalog-row" key={s.id}>
                        <h3>{s.label}</h3>
                        <p>{s.description}</p>
                        <small>{s.shape} · Target: {s.target} · {s.provenance_note}</small>
                        <button className="text-action" onClick={() => p.onSelectScenario(s)}>Inspect dataset</button>
                      </article>
                    ))}
                </details>
              </div>
            )}

            {/* Tab 2: Live Providers */}
            {hubTab === 'live' && (
              <div>
                <label>Remote Provider</label>
                <div className="execution-mode-choices" style={{ marginBottom: '10px' }}>
                  {(['huggingface', 'openml', 'uci', 'kaggle'] as const).map(pr => (
                    <button
                      key={pr}
                      type="button"
                      className={provider === pr ? 'primary' : 'tonal'}
                      onClick={() => setProviderAndPreset(pr)}
                    >
                      {pr === 'huggingface' ? 'Hugging Face' : pr.toUpperCase()}
                    </button>
                  ))}
                </div>
                <div className="configuration-fields">
                  <label className="stacked-label">
                    Dataset identifier
                    <input
                      value={datasetId}
                      onChange={e => setDatasetId(e.target.value)}
                      placeholder="e.g. scikit-learn/adult-census-income"
                    />
                  </label>
                  <label className="stacked-label">
                    Target column (optional)
                    <input
                      value={target}
                      onChange={e => setTarget(e.target.value)}
                      placeholder="Auto-detect"
                    />
                  </label>
                  <label className="stacked-label">
                    Revision / split
                    <input
                      value={revision}
                      onChange={e => setRevision(e.target.value)}
                      placeholder="main / default"
                    />
                  </label>
                </div>
                <div className="inline-actions" style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    className="tonal"
                    disabled={!!liveBusy || !datasetId.trim()}
                    onClick={handleResolveLive}
                  >
                    Resolve dataset
                  </button>
                  <button
                    type="button"
                    className="tonal"
                    disabled={!!liveBusy || !datasetId.trim()}
                    onClick={handlePrecertifyLive}
                  >
                    Run pre-certification
                  </button>
                  <button
                    type="button"
                    className="primary"
                    disabled={!datasetId.trim()}
                    onClick={handleSelectLiveForExecution}
                  >
                    Use for Execution
                  </button>
                </div>
                {liveBusy && <p role="status" className="field-help">{liveBusy}</p>}
                {liveError && <p className="attention-note" role="alert">{liveError}</p>}
                {contract && (
                  <details open style={{ marginTop: '8px' }}>
                    <summary>Resolved Dataset Contract</summary>
                    <DataValue value={{
                      dataset_id: contract.dataset_id,
                      provider: contract.provider,
                      target_column: contract.target_column,
                      shape: `${contract.rows_declared ?? 'unbounded'} rows × ${contract.features?.length ?? 0} features`,
                      features_sample: (contract.features ?? []).slice(0, 8).map((f: any) => `${f.name} (${f.data_type})`),
                    }} />
                  </details>
                )}
                {report && (
                  <details open style={{ marginTop: '8px' }}>
                    <summary>Pre-certification: {report.overall_status}</summary>
                    <DataValue value={{
                      status: report.overall_status,
                      schema_validity: report.schema_validity,
                      target_validity: report.target_validity,
                      feature_roles: report.feature_roles,
                      split_validity: report.split_validity,
                      evaluated_sample_rows: report.sample_rows,
                    }} />
                  </details>
                )}
              </div>
            )}

            {/* Tab 3: Local File */}
            {hubTab === 'local' && (
              <div>
                <label className="stacked-label">
                  Local File Path (CSV / Parquet)
                  <input
                    value={localPath}
                    onChange={e => setLocalPath(e.target.value)}
                    placeholder="e.g. data/credit_risk.csv"
                  />
                </label>
                <div className="inline-actions" style={{ marginTop: '8px' }}>
                  <button
                    type="button"
                    className="primary"
                    disabled={!localPath.trim()}
                    onClick={handleSelectLocalForExecution}
                  >
                    Use for Execution
                  </button>
                </div>
              </div>
            )}

            {/* Execution Dataset indicator */}
            {p.context && (
              <div className="catalog-row" style={{ marginTop: '12px', borderLeft: '3px solid var(--accent, #3b82f6)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <CheckCircle size={14} color="#10b981" />
                  <small style={{ fontWeight: 600 }}>BOUND EXECUTION DATASET</small>
                </div>
                <h3 style={{ margin: '4px 0' }}>{context?.label || p.context}</h3>
                <small>{context?.provenance} · Target: {context?.target} · Seed: {context?.seed}</small>
              </div>
            )}

            {inspectedScenario && (
              <p className="attention-note">
                Inspecting {p.selectedScenario!.label}. This scenario has no execution binding.{' '}
                <button className="text-action" onClick={() => p.setContext(p.context!)}>Return to execution dataset</button>
              </p>
            )}
          </div>
        </div>

        {/* Execution settings */}
        <details>
          <summary>Execution settings</summary>
          <div className="configuration-fields">
            <label className="stacked-label">
              Seed
              <input
                type="number"
                min="0"
                max="4294967295"
                step="1"
                value={p.seed ?? ''}
                placeholder="Backend default"
                onChange={e => p.onSeedChange?.(e.target.value === '' ? undefined : Number(e.target.value))}
              />
            </label>
            {p.workflow === 'hyperparameter_tuning' && (
              <label className="stacked-label">
                Trial budget
                <input
                  type="number"
                  min="5"
                  max="30"
                  step="1"
                  placeholder="Backend default: 10"
                  value={p.trials ?? ''}
                  onChange={e => p.onTrialsChange?.(e.target.value === '' ? undefined : Number(e.target.value))}
                />
              </label>
            )}
          </div>
          <p className="quiet">The run applies these controls. Resolved configuration is recorded in execution evidence.</p>
        </details>

        {/* Engineering Controls */}
        {p.manifest && (
          <EngineeringControls
            manifest={p.manifest}
            workflow={p.workflow}
            parameters={p.parameters ?? {}}
            onChange={p.onParametersChange ?? (() => {})}
            setContext={p.setContext}
          />
        )}
      </div>

      {mismatchedAlgorithm && (
        <p className="attention-note">Select a recommender algorithm and its matching dataset before building the plan.</p>
      )}

      {/* Action buttons */}
      <div className="define-actions">
        <button className="primary" disabled={!ready} onClick={p.onPlan}>
          {p.busy ? 'Preparing…' : 'Build plan'}
          <ArrowRight size={15} />
        </button>
        <button className="text-action" onClick={p.onOpenTestCatalog}>
          Capability & evaluation catalog
        </button>
      </div>

      {/* Build & Run plan */}
      {planCurrent && (
        <section className="review-plan">
          <div className="eyebrow">BUILD & RUN PLAN</div>
          <h2>Inspect before execution</h2>
          {p.plan!.agentProposal && (
            <section>
              <h3>Backend proposal</h3>
              <DataValue value={p.plan!.agentProposal} />
              <p className="quiet">This is a backend template proposal. It has not changed your selected configuration. Edit the controls above to override it.</p>
            </section>
          )}
          <p>{context?.label || p.context} · {p.plan!.plan.length} stages</p>
          <ol className="planned-milestones">
            {p.plan!.plan.map((s, i) => (
              <li key={s.id}>
                <span>{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <strong>{s.label}</strong>
                  <p>{s.description}</p>
                </div>
              </li>
            ))}
          </ol>
          <DataValue value={{
            execution_mode: p.executionMode,
            context_id: p.context,
            parameters: p.parameters,
            seed: p.seed ?? 'Backend default',
            ...(p.workflow === 'hyperparameter_tuning' ? { trials: p.trials ?? 'Backend default: 10' } : {}),
          }} />
          {p.plan!.warnings?.map((v, i) => <p key={i}>{v}</p>)}
          <button className="primary" disabled={!ready} onClick={p.onStart}>
            Accept & execute plan
            <ArrowRight size={15} />
          </button>
        </section>
      )}

      <footer className="define-footer">
        Agents propose / reason. Deterministic tools calculate. EvidenceRecords prove.
      </footer>
    </div>
  )
}
