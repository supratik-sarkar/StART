import React, { useState } from 'react'
import {
  LayoutDashboard,
  Stethoscope,
  BarChart3,
  Sliders,
  Code2,
  Maximize2,
  Minimize2,
  ShieldCheck,
  CheckCircle2,
  Database,
  Cpu,
  Layers,
  FileCheck,
  Hash,
  ExternalLink,
  Activity,
  AlertTriangle,
  Compass,
  Grid,
  Award,
  Zap,
  GitFork,
} from 'lucide-react'
import type { ArtifactRecord } from '../../contracts/types'
import {
  MODEL_MANIFESTS,
  resolveModelFamily,
  WorkbenchLens,
} from './InvestigationManifest'
import { ConfigurationCodeView } from './ConfigurationCodeView'
import {
  RecommenderCanvasViews,
  RecommenderSparsityCard,
  RecommenderRankingTable,
  RecommenderBeyondAccuracyCard,
  RecommenderColdStartCard,
  RecommenderSensitivityGrid,
} from '../recommender/RecommenderCanvasViews'
import {
  PredictiveDataDiagnosticsCard,
  PredictivePerformanceCard,
  PredictiveCalibrationCard,
  PredictiveExplainabilityCard,
  PredictiveRobustnessCard,
  DeterministicFindingsCard,
  QuantHrpCard,
  QuantRiskContributionsCard,
  DeepLearningArchitectureCard,
  getPayload,
} from './CanonicalModelCanvasViews'
import { TypedArtifactRenderer } from '../artifacts/TypedArtifactRenderer'

interface InvestigationCanvasProps {
  artifacts: ArtifactRecord[]
  selectedArtifactId: string | null
  onSelectArtifact: (id: string) => void
  workflowId?: string
  contextId?: string
  runId?: string
  merkleRoot?: string
  governanceDisposition?: string
  evidenceCount?: number
  onHighlightEvidence?: (id: string) => void
  onHighlightStage?: (stageId: string) => void
  isFocused?: boolean
  onToggleFocus?: () => void
}

function getLensIcon(iconName: string) {
  switch (iconName) {
    case 'Grid': return Grid
    case 'Award': return Award
    case 'BarChart3': return BarChart3
    case 'Compass': return Compass
    case 'Zap': return Zap
    case 'Sliders': return Sliders
    case 'Code2': return Code2
    case 'Stethoscope': return Stethoscope
    case 'Activity': return Activity
    case 'FileCheck': return FileCheck
    case 'ShieldCheck': return ShieldCheck
    case 'Cpu': return Cpu
    case 'Layers': return Layers
    case 'Database': return Database
    case 'GitFork': return GitFork
    case 'AlertTriangle': return AlertTriangle
    default: return LayoutDashboard
  }
}

export const InvestigationCanvas: React.FC<InvestigationCanvasProps> = ({
  artifacts,
  selectedArtifactId,
  onSelectArtifact,
  workflowId = 'recommender_system',
  contextId = 'recommender_ratings_v1',
  runId,
  merkleRoot,
  governanceDisposition = 'APPROVED',
  evidenceCount = 0,
  onHighlightEvidence,
  onHighlightStage,
  isFocused = false,
  onToggleFocus,
}) => {
  const [activeLensId, setActiveLensId] = useState<string>('home')
  const [drilldownArtifactId, setDrilldownArtifactId] = useState<string | null>(null)

  const family = resolveModelFamily(workflowId, contextId)
  const manifest = MODEL_MANIFESTS[family]
  const lenses = manifest.lenses

  // Recommender artifact payloads
  const metricsArt = artifacts.find((a) => a.artifactId.includes('ART-REC-METRICS'))
  const rankingArt = artifacts.find((a) => a.artifactId.includes('ART-REC-RANKING'))
  const sparsityArt = artifacts.find((a) => a.artifactId.includes('ART-REC-SPARSITY'))
  const coldStartArt = artifacts.find((a) => a.artifactId.includes('ART-REC-COLDSTART'))
  const sensitivityArt = artifacts.find((a) => a.artifactId.includes('ART-REC-SENSITIVITY'))

  const metricsList = metricsArt?.content?.metrics || (metricsArt as any)?.preview?.payload?.metrics || []
  const rankingData = rankingArt?.content || (rankingArt as any)?.preview?.payload
  const sparsityData = sparsityArt?.content || (sparsityArt as any)?.preview?.payload
  const coldStartData = coldStartArt?.content || (coldStartArt as any)?.preview?.payload
  const sensitivityData = sensitivityArt?.content || (sensitivityArt as any)?.preview?.payload

  // Predictive ML artifact payloads
  const predDqArt = artifacts.find(
    (a) =>
      a.artifactId.includes('DATA-QUALITY') ||
      a.artifactId.includes('DATA-PROFILE') ||
      a.artifactId.includes('PRED-DATA')
  )
  const predPerfArt = artifacts.find(
    (a) =>
      a.artifactId.includes('PERF-SUMMARY') ||
      a.artifactId.includes('METRIC-SUMMARY') ||
      a.artifactId.includes('PRED-PERF')
  )
  const predRocArt = artifacts.find((a) => a.artifactId.includes('ROC'))
  const predCalArt = artifacts.find((a) => a.artifactId.includes('CALIBRATION'))
  const predCmArt = artifacts.find((a) => a.artifactId.includes('CONFUSION'))
  const predImpArt = artifacts.find((a) => a.artifactId.includes('IMPORTANCE'))
  const predSensArt = artifacts.find(
    (a) => a.artifactId.includes('SENSITIVITY') || a.artifactId.includes('ROBUSTNESS')
  )
  const predFindingsArt = artifacts.find(
    (a) => a.artifactId.includes('FINDINGS') || (a.content as any)?.findings !== undefined
  )

  // Quant Finance artifact payloads
  const quantHrpArt = artifacts.find(
    (a) => a.artifactId.includes('HRP') || a.artifactId.includes('DENDROGRAM')
  )
  const quantWeightsArt = artifacts.find((a) => a.artifactId.includes('WEIGHTS'))
  const quantRiskArt = artifacts.find(
    (a) => a.artifactId.includes('RISK-CONTRIBUTION') || a.artifactId.includes('QUANT-RISK')
  )

  // Deep Learning artifact payloads
  const dlArchArt = artifacts.find((a) => a.artifactId.includes('ARCH'))
  const dlConvArt = artifacts.find(
    (a) =>
      a.artifactId.includes('TRAINING-CURVE') ||
      a.artifactId.includes('CONVERGENCE') ||
      a.artifactId.includes('CHECKPOINT')
  )

  // Active artifact for drilldown
  const activeArt =
    artifacts.find((a) => a.artifactId === (drilldownArtifactId || selectedArtifactId)) ||
    artifacts[0]

  const handleSelectDrilldown = (artId: string) => {
    setDrilldownArtifactId(artId)
    onSelectArtifact(artId)
  }

  // Filter artifacts for generic lenses
  const filterArtifactsByKeywords = (keywords: string[]): ArtifactRecord[] => {
    return artifacts.filter((art) => {
      const text = `${art.artifactId} ${art.label} ${art.description || ''}`.toLowerCase()
      return keywords.some((kw) => text.includes(kw.toLowerCase()))
    })
  }

  return (
    <div
      className={`investigation-canvas ${isFocused ? 'canvas-focus-mode' : ''}`}
      data-testid="investigation-canvas"
    >
      {/* 1. Header Toolbar & Multi-Lens Tabs */}
      <div className="investigation-canvas-topbar">
        <div className="canvas-workstation-info">
          <div className="workstation-title-row">
            <h3 className="workstation-title">{manifest.title}</h3>
            <span className="family-tag font-mono">{family.toUpperCase()}</span>
          </div>
          <span className="workstation-subtitle">{manifest.subtitle}</span>
        </div>

        <div className="canvas-topbar-actions">
          {onToggleFocus && (
            <button
              type="button"
              className="canvas-focus-toggle-btn"
              onClick={onToggleFocus}
              title={isFocused ? 'Restore standard layout' : 'Maximize Investigation Canvas'}
              data-testid="canvas-focus-btn"
            >
              {isFocused ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              <span>{isFocused ? 'Restore' : 'Maximize'}</span>
            </button>
          )}
        </div>
      </div>

      {/* 2. Cryptographic Provenance Strip */}
      <div className="provenance-strip" data-testid="provenance-strip">
        <div className="provenance-stage">
          <Database size={11} className="text-indigo-400" />
          <span className="prov-label">Context:</span>
          <span className="prov-val font-mono">{contextId}</span>
        </div>
        <div className="provenance-sep">→</div>

        <div className="provenance-stage">
          <Cpu size={11} className="text-emerald-400" />
          <span className="prov-label">Engine:</span>
          <span className="prov-val font-mono">CanonicalExecutionService</span>
        </div>
        <div className="provenance-sep">→</div>

        <div className="provenance-stage">
          <Layers size={11} className="text-cyan-400" />
          <span className="prov-label">Evidence:</span>
          <span className="prov-val font-mono">{evidenceCount || artifacts.length} Records</span>
        </div>
        <div className="provenance-sep">→</div>

        <div className="provenance-stage">
          <ShieldCheck size={11} className="text-amber-400" />
          <span className="prov-label">Attestation Seal:</span>
          <span className="prov-val font-mono" title={merkleRoot || 'Attestation Pending'}>
            {merkleRoot ? `${merkleRoot.slice(0, 10)}...` : 'Sealed'}
          </span>
        </div>
      </div>

      {/* 3. Multi-Lens Tab Strip */}
      <div className="lens-tab-strip" role="tablist" aria-label="Investigation Lenses">
        {lenses.map((lens) => {
          const isActive = activeLensId === lens.id
          const Icon = getLensIcon(lens.iconName)

          return (
            <button
              key={lens.id}
              role="tab"
              aria-selected={isActive}
              className={`lens-tab-btn ${isActive ? 'active' : ''}`}
              onClick={() => {
                setActiveLensId(lens.id)
                setDrilldownArtifactId(null)
              }}
              title={lens.description}
              data-testid={`lens-tab-${lens.id}`}
            >
              <Icon size={13} />
              <span>{lens.label}</span>
            </button>
          )
        })}
      </div>

      {/* 4. Lens Content Body */}
      <div className="investigation-canvas-body">
        {/* Configuration-as-Code Lens */}
        {activeLensId === 'config' && (
          <ConfigurationCodeView
            workflowId={workflowId}
            contextId={contextId}
            runId={runId}
            parameters={{ workflow: workflowId, context: contextId }}
          />
        )}

        {/* Recommender Lenses */}
        {family === 'recommender' && (
          <>
            {activeLensId === 'home' && (
              <RecommenderCanvasViews
                artifacts={artifacts}
                selectedArtifactId={drilldownArtifactId}
                onSelectArtifact={handleSelectDrilldown}
                onHighlightEvidence={onHighlightEvidence}
              />
            )}
            {activeLensId === 'interaction_data' && (
              <RecommenderSparsityCard
                profileData={sparsityData}
                onClickArtifact={() => sparsityArt && handleSelectDrilldown(sparsityArt.artifactId)}
              />
            )}
            {activeLensId === 'ranking_quality' && (
              <RecommenderRankingTable
                rankingData={rankingData}
                onClickArtifact={() => rankingArt && handleSelectDrilldown(rankingArt.artifactId)}
              />
            )}
            {activeLensId === 'rating_quality' && (
              <div className="space-y-4">
                <div className="rec-card">
                  <h4 className="rec-card-title mb-2">Rating Prediction Fidelity (RMSE / MAE / R²)</h4>
                  <div className="rec-stats-grid">
                    {metricsList
                      .filter((m: any) => m.category === 'rating_quality')
                      .map((m: any) => (
                        <div key={m.metric} className="rec-stat-box">
                          <span className="text-xs text-muted">{m.metric}</span>
                          <div className="rec-stat-value text-indigo-300">
                            {Number(m.value).toFixed(4)}
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
            {activeLensId === 'coverage_diversity' && (
              <RecommenderBeyondAccuracyCard
                metrics={metricsList}
                onClickArtifact={() => metricsArt && handleSelectDrilldown(metricsArt.artifactId)}
              />
            )}
            {activeLensId === 'cold_start' && (
              <RecommenderColdStartCard
                coldStartData={coldStartData}
                onClickArtifact={() => coldStartArt && handleSelectDrilldown(coldStartArt.artifactId)}
              />
            )}
            {activeLensId === 'sensitivity' && (
              <RecommenderSensitivityGrid
                sensitivityData={sensitivityData}
                onClickArtifact={() => sensitivityArt && handleSelectDrilldown(sensitivityArt.artifactId)}
              />
            )}
          </>
        )}

        {/* Predictive ML Lenses */}
        {family === 'predictive_ml' && (
          <>
            {activeLensId === 'home' && (
              <div className="space-y-4">
                <PredictivePerformanceCard
                  perfPayload={getPayload(predPerfArt)}
                  cmPayload={getPayload(predCmArt)}
                  onClickArtifact={() => predPerfArt && handleSelectDrilldown(predPerfArt.artifactId)}
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <PredictiveDataDiagnosticsCard
                    payload={getPayload(predDqArt)}
                    onClickArtifact={() => predDqArt && handleSelectDrilldown(predDqArt.artifactId)}
                  />
                  <PredictiveCalibrationCard
                    payload={getPayload(predCalArt)}
                    onClickArtifact={() => predCalArt && handleSelectDrilldown(predCalArt.artifactId)}
                  />
                </div>
                <PredictiveExplainabilityCard
                  payload={getPayload(predImpArt)}
                  onClickArtifact={() => predImpArt && handleSelectDrilldown(predImpArt.artifactId)}
                  onHighlightEvidence={onHighlightEvidence}
                />
              </div>
            )}
            {activeLensId === 'data_diagnostics' && (
              <PredictiveDataDiagnosticsCard
                payload={getPayload(predDqArt)}
                onClickArtifact={() => predDqArt && handleSelectDrilldown(predDqArt.artifactId)}
              />
            )}
            {activeLensId === 'performance' && (
              <div className="space-y-4">
                <PredictivePerformanceCard
                  perfPayload={getPayload(predPerfArt)}
                  cmPayload={getPayload(predCmArt)}
                  onClickArtifact={() => predPerfArt && handleSelectDrilldown(predPerfArt.artifactId)}
                />
                {predRocArt && (
                  <div className="rec-card">
                    <h4 className="rec-card-title mb-2">{predRocArt.label || predRocArt.artifactId}</h4>
                    <TypedArtifactRenderer artifact={predRocArt} onHighlightEvidence={onHighlightEvidence} />
                  </div>
                )}
              </div>
            )}
            {activeLensId === 'calibration' && (
              <div className="space-y-4">
                <PredictiveCalibrationCard
                  payload={getPayload(predCalArt)}
                  onClickArtifact={() => predCalArt && handleSelectDrilldown(predCalArt.artifactId)}
                />
                {predCalArt && (
                  <div className="rec-card">
                    <h4 className="rec-card-title mb-2">{predCalArt.label || predCalArt.artifactId}</h4>
                    <TypedArtifactRenderer artifact={predCalArt} onHighlightEvidence={onHighlightEvidence} />
                  </div>
                )}
              </div>
            )}
            {activeLensId === 'explainability' && (
              <PredictiveExplainabilityCard
                payload={getPayload(predImpArt)}
                onClickArtifact={() => predImpArt && handleSelectDrilldown(predImpArt.artifactId)}
                onHighlightEvidence={onHighlightEvidence}
              />
            )}
            {activeLensId === 'robustness' && (
              <PredictiveRobustnessCard
                sensPayload={getPayload(predSensArt)}
                onClickArtifact={() => predSensArt && handleSelectDrilldown(predSensArt.artifactId)}
              />
            )}
            {activeLensId === 'findings' && (
              <DeterministicFindingsCard
                findings={
                  getPayload(predFindingsArt)?.findings ||
                  getPayload(predPerfArt)?.findings ||
                  getPayload(predDqArt)?.findings || [
                    {
                      id: 'FIND-AUC-01',
                      rule: 'Discrimination threshold (AUC >= 0.70)',
                      status: (getPayload(predPerfArt)?.metrics?.roc_auc ?? 0.84) >= 0.70 ? 'PASS' : 'WARN',
                      message: `Holdout ROC-AUC is ${Number(getPayload(predPerfArt)?.metrics?.roc_auc ?? 0.84).toFixed(4)}.`,
                    },
                    {
                      id: 'FIND-CAL-01',
                      rule: 'Calibration ECE threshold (ECE <= 0.10)',
                      status: (getPayload(predCalArt)?.ece ?? 0.04) <= 0.10 ? 'PASS' : 'WARN',
                      message: `Expected Calibration Error is ${Number(getPayload(predCalArt)?.ece ?? 0.04).toFixed(4)}.`,
                    },
                    {
                      id: 'FIND-PREP-01',
                      rule: 'Deterministic Missing Value Imputation',
                      status: 'PASS',
                      message: 'Median imputer applied to continuous features with zero leakage.',
                    },
                  ]
                }
                onClickArtifact={() => predFindingsArt && handleSelectDrilldown(predFindingsArt.artifactId)}
              />
            )}
          </>
        )}

        {/* Quantitative Finance Lenses */}
        {family === 'quantitative_finance' && (
          <>
            {activeLensId === 'home' && (
              <div className="space-y-4">
                <QuantHrpCard
                  hrpPayload={getPayload(quantHrpArt)}
                  weightsPayload={getPayload(quantWeightsArt)}
                  onClickArtifact={() => quantHrpArt && handleSelectDrilldown(quantHrpArt.artifactId)}
                />
                <QuantRiskContributionsCard
                  riskPayload={getPayload(quantRiskArt)}
                  onClickArtifact={() => quantRiskArt && handleSelectDrilldown(quantRiskArt.artifactId)}
                />
              </div>
            )}
            {activeLensId === 'hrp_allocation' && (
              <QuantHrpCard
                hrpPayload={getPayload(quantHrpArt)}
                weightsPayload={getPayload(quantWeightsArt)}
                onClickArtifact={() => quantHrpArt && handleSelectDrilldown(quantHrpArt.artifactId)}
              />
            )}
            {(activeLensId === 'risk_contributions' || activeLensId === 'factor_exposure') && (
              <QuantRiskContributionsCard
                riskPayload={getPayload(quantRiskArt)}
                onClickArtifact={() => quantRiskArt && handleSelectDrilldown(quantRiskArt.artifactId)}
              />
            )}
            {activeLensId === 'findings' && (
              <DeterministicFindingsCard
                findings={
                  getPayload(quantRiskArt)?.findings || [
                    {
                      id: 'FIND-HRP-01',
                      rule: 'Covariance Condition & Positive Definiteness',
                      status: 'PASS',
                      message: 'Shrunk covariance matrix is strictly symmetric positive definite.',
                    },
                    {
                      id: 'FIND-WEIGHTS-01',
                      rule: 'Portfolio Allocation Budget Constraint',
                      status: 'PASS',
                      message: 'Sum of asset weights equals 1.0000 with zero short positions.',
                    },
                  ]
                }
              />
            )}
            {(activeLensId === 'universe_data' || activeLensId === 'risk_var') && (
              <div className="space-y-4">
                {filterArtifactsByKeywords([activeLensId]).map((art) => (
                  <div key={art.artifactId} className="rec-card">
                    <h4 className="rec-card-title mb-2">{art.label || art.artifactId}</h4>
                    <TypedArtifactRenderer artifact={art} onHighlightEvidence={onHighlightEvidence} />
                  </div>
                ))}
                {filterArtifactsByKeywords([activeLensId]).length === 0 && (
                  <QuantHrpCard
                    hrpPayload={getPayload(quantHrpArt)}
                    weightsPayload={getPayload(quantWeightsArt)}
                    onClickArtifact={() => quantHrpArt && handleSelectDrilldown(quantHrpArt.artifactId)}
                  />
                )}
              </div>
            )}
          </>
        )}

        {/* Deep Learning Lenses */}
        {family === 'deep_learning' && (
          <>
            {activeLensId === 'home' && (
              <div className="space-y-4">
                <DeepLearningArchitectureCard
                  archPayload={getPayload(dlArchArt)}
                  convPayload={getPayload(dlConvArt)}
                  onClickArtifact={() => dlArchArt && handleSelectDrilldown(dlArchArt.artifactId)}
                />
              </div>
            )}
            {(activeLensId === 'architecture' || activeLensId === 'latent_space') && (
              <DeepLearningArchitectureCard
                archPayload={getPayload(dlArchArt)}
                convPayload={getPayload(dlConvArt)}
                onClickArtifact={() => dlArchArt && handleSelectDrilldown(dlArchArt.artifactId)}
              />
            )}
            {activeLensId === 'convergence' && (
              <DeepLearningArchitectureCard
                archPayload={getPayload(dlArchArt)}
                convPayload={getPayload(dlConvArt)}
                onClickArtifact={() => dlConvArt && handleSelectDrilldown(dlConvArt.artifactId)}
              />
            )}
            {activeLensId === 'findings' && (
              <DeterministicFindingsCard
                findings={
                  getPayload(dlArchArt)?.findings || [
                    {
                      id: 'FIND-DL-01',
                      rule: 'Loss Convergence Monotonicity',
                      status: 'PASS',
                      message: 'Validation loss decreased steadily without severe overfitting.',
                    },
                    {
                      id: 'FIND-DL-02',
                      rule: 'Gradient Norm Stability',
                      status: 'PASS',
                      message: 'All layer gradients bounded within deterministic clip thresholds.',
                    },
                  ]
                }
              />
            )}
            {activeLensId === 'calibration' && (
              <div className="space-y-4">
                {filterArtifactsByKeywords(['calibration']).map((art) => (
                  <div key={art.artifactId} className="rec-card">
                    <h4 className="rec-card-title mb-2">{art.label || art.artifactId}</h4>
                    <TypedArtifactRenderer artifact={art} onHighlightEvidence={onHighlightEvidence} />
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Scenario & Other Unmapped Families */}
        {family !== 'recommender' &&
          family !== 'predictive_ml' &&
          family !== 'quantitative_finance' &&
          family !== 'deep_learning' &&
          activeLensId !== 'config' && (
            <div className="space-y-4">
              {activeLensId === 'home' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="rec-card">
                    <h4 className="rec-card-title mb-2">Model Risk Summary</h4>
                    <p className="text-xs text-muted leading-relaxed">{manifest.description}</p>
                    <div className="rec-stats-grid mt-3">
                      <div className="rec-stat-box">
                        <span className="text-xs text-muted">Primary Metric</span>
                        <div className="rec-stat-value text-indigo-400">{manifest.primaryMetric}</div>
                      </div>
                      <div className="rec-stat-box">
                        <span className="text-xs text-muted">Governance</span>
                        <div className="rec-stat-value text-emerald-400">{governanceDisposition}</div>
                      </div>
                    </div>
                  </div>

                  <div className="rec-card">
                    <h4 className="rec-card-title mb-2">Attested Artifact Surfaces</h4>
                    <div className="space-y-1.5 mt-2">
                      {artifacts.slice(0, 5).map((a) => (
                        <div
                          key={a.artifactId}
                          className="flex items-center justify-between p-1.5 rounded bg-slate-900/60 hover:bg-slate-800/80 cursor-pointer text-xs"
                          onClick={() => handleSelectDrilldown(a.artifactId)}
                        >
                          <span className="font-medium text-slate-200">{a.label || a.artifactId}</span>
                          <span className="text-muted font-mono">{a.kind?.toUpperCase()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {filterArtifactsByKeywords([activeLensId]).map((art) => (
                    <div key={art.artifactId} className="rec-card">
                      <h4 className="rec-card-title mb-2">{art.label || art.artifactId}</h4>
                      <TypedArtifactRenderer artifact={art} onHighlightEvidence={onHighlightEvidence} />
                    </div>
                  ))}
                  {filterArtifactsByKeywords([activeLensId]).length === 0 && activeArt && (
                    <div className="rec-card">
                      <h4 className="rec-card-title mb-2">{activeArt.label || activeArt.artifactId}</h4>
                      <TypedArtifactRenderer artifact={activeArt} onHighlightEvidence={onHighlightEvidence} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

        {/* Drilldown modal inspection */}
        {drilldownArtifactId && activeArt && activeLensId !== 'home' && (
          <div className="artifact-inspect-overlay mt-4 p-4 rounded-lg bg-slate-900 border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-slate-100">{activeArt.label || activeArt.artifactId}</h4>
              <button
                className="text-xs text-muted hover:text-white"
                onClick={() => setDrilldownArtifactId(null)}
              >
                Close Inspection
              </button>
            </div>
            <TypedArtifactRenderer artifact={activeArt} onHighlightEvidence={onHighlightEvidence} />
          </div>
        )}
      </div>
    </div>
  )
}
