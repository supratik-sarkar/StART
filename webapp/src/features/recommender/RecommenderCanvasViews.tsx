import React from 'react'
import {
  Grid,
  Users,
  Package,
  Activity,
  Layers,
  TrendingDown,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Sparkles,
  Award,
  Compass,
  Zap,
} from 'lucide-react'
import type { ArtifactRecord } from '../../contracts/types'

interface RecommenderCanvasProps {
  artifacts: ArtifactRecord[]
  selectedArtifactId?: string | null
  onSelectArtifact?: (id: string) => void
  onHighlightEvidence?: (evidenceId: string) => void
}

/**
 * 1. Interaction Sparsity & Density Card
 */
export const RecommenderSparsityCard: React.FC<{
  profileData?: any
  onClickArtifact?: () => void
}> = ({ profileData, onClickArtifact }) => {
  if (!profileData) {
    return (
      <div className="rec-card rec-sparsity-card empty-card" data-testid="rec-sparsity-card">
        <div className="rec-card-header">
          <div className="flex items-center gap-2">
            <Grid size={15} className="text-indigo-400" />
            <h4 className="rec-card-title">Matrix Sparsity & Density</h4>
          </div>
        </div>
        <p className="text-xs text-muted">Awaiting dataset profiling artifact...</p>
      </div>
    )
  }

  const users = profileData.users ?? 0
  const items = profileData.items ?? 0
  const interactions = profileData.interactions ?? 0
  const sparsityRatio = profileData.sparsity_ratio ?? 0
  const densityPercent = profileData.density_percent ?? 0
  const interactionsPerUser = profileData.interactions_per_user ?? 0
  const interactionsPerItem = profileData.interactions_per_item ?? 0
  const warnings = profileData.warnings || []

  return (
    <div className="rec-card rec-sparsity-card" data-testid="rec-sparsity-card" onClick={onClickArtifact}>
      <div className="rec-card-header">
        <div className="flex items-center gap-2">
          <Grid size={15} className="text-indigo-400" />
          <h4 className="rec-card-title">Matrix Sparsity & Density</h4>
        </div>
        <span className="rec-badge-density">
          {densityPercent.toFixed(2)}% Dense
        </span>
      </div>

      <div className="rec-stats-grid">
        <div className="rec-stat-box">
          <div className="flex items-center gap-1.5 text-muted text-xs">
            <Users size={12} />
            <span>Users</span>
          </div>
          <div className="rec-stat-value">{users.toLocaleString()}</div>
        </div>

        <div className="rec-stat-box">
          <div className="flex items-center gap-1.5 text-muted text-xs">
            <Package size={12} />
            <span>Items</span>
          </div>
          <div className="rec-stat-value">{items.toLocaleString()}</div>
        </div>

        <div className="rec-stat-box">
          <div className="flex items-center gap-1.5 text-muted text-xs">
            <Activity size={12} />
            <span>Interactions</span>
          </div>
          <div className="rec-stat-value">{interactions.toLocaleString()}</div>
        </div>

        <div className="rec-stat-box">
          <div className="flex items-center gap-1.5 text-muted text-xs">
            <Layers size={12} />
            <span>Sparsity</span>
          </div>
          <div className="rec-stat-value">{(sparsityRatio * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="rec-sparsity-progress">
        <div className="flex justify-between text-xs text-muted mb-1">
          <span>Interaction Fill Rate</span>
          <span>{densityPercent.toFixed(3)}%</span>
        </div>
        <div className="progress-track-mini">
          <div
            className="progress-fill-mini"
            style={{ width: `${Math.min(100, Math.max(2, densityPercent))}%` }}
          />
        </div>
      </div>

      <div className="rec-meta-row flex justify-between text-xs text-muted pt-2 border-t border-slate-700/40">
        <span>Avg/User: <strong className="text-slate-200">{interactionsPerUser}</strong></span>
        <span>Avg/Item: <strong className="text-slate-200">{interactionsPerItem}</strong></span>
      </div>

      {warnings.length > 0 && (
        <div className="rec-warnings-box mt-2">
          {warnings.map((w: string, i: number) => (
            <div key={i} className="flex items-center gap-1.5 text-amber-400 text-xs">
              <AlertTriangle size={11} />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * 2. Top-K Ranking Performance Table
 */
export const RecommenderRankingTable: React.FC<{
  rankingData?: any
  onClickArtifact?: () => void
}> = ({ rankingData, onClickArtifact }) => {
  if (!rankingData) {
    return (
      <div className="rec-card rec-ranking-card empty-card" data-testid="rec-ranking-card">
        <div className="rec-card-header">
          <div className="flex items-center gap-2">
            <Award size={15} className="text-emerald-400" />
            <h4 className="rec-card-title">Top-K Ranking Evaluation</h4>
          </div>
        </div>
        <p className="text-xs text-muted">Awaiting Top-K ranking artifact...</p>
      </div>
    )
  }

  const kEvaluations: any[] = rankingData.k_evaluations || []
  const mrr = rankingData.mrr ?? 0

  return (
    <div className="rec-card rec-ranking-card" data-testid="rec-ranking-card" onClick={onClickArtifact}>
      <div className="rec-card-header">
        <div className="flex items-center gap-2">
          <Award size={15} className="text-emerald-400" />
          <h4 className="rec-card-title">Top-K Ranking Evaluation</h4>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Mean Reciprocal Rank (MRR):</span>
          <span className="rec-badge-highlight font-mono">{mrr.toFixed(4)}</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="rec-table" data-testid="rec-ranking-table">
          <thead>
            <tr>
              <th>Cutoff (K)</th>
              <th>NDCG@K</th>
              <th>Recall@K</th>
              <th>Precision@K</th>
              <th>Hit Rate@K</th>
              <th>MAP@K</th>
            </tr>
          </thead>
          <tbody>
            {kEvaluations.map((row) => (
              <tr key={row.cutoff_k}>
                <td className="font-semibold text-slate-200">K = {row.cutoff_k}</td>
                <td>
                  <span className="rec-val-pill text-emerald-400 bg-emerald-950/40">
                    {(row.ndcg ?? 0).toFixed(4)}
                  </span>
                </td>
                <td>
                  <span className="font-mono text-slate-300">{(row.recall ?? 0).toFixed(4)}</span>
                </td>
                <td>
                  <span className="font-mono text-slate-300">{(row.precision ?? 0).toFixed(4)}</span>
                </td>
                <td>
                  <span className="font-mono text-slate-300">{(row.hit_rate ?? 0).toFixed(4)}</span>
                </td>
                <td>
                  <span className="font-mono text-slate-400">{(row.map ?? 0).toFixed(4)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-muted mt-2">
        <span>Evaluated with exact leave-one-out and user-stratified test holdout sets.</span>
      </div>
    </div>
  )
}

/**
 * 3. Beyond-Accuracy & Diversity Card
 */
export const RecommenderBeyondAccuracyCard: React.FC<{
  metrics?: any[]
  onClickArtifact?: () => void
}> = ({ metrics = [], onClickArtifact }) => {
  const findVal = (name: string): number => {
    const item = metrics.find((m) => m.metric === name)
    return item ? Number(item.value) : 0
  }

  const catalogCoverage = findVal('Catalog Coverage')
  const userCoverage = findVal('User Coverage')
  const novelty = findVal('Novelty')
  const popularityBias = findVal('Popularity Bias')

  return (
    <div className="rec-card rec-beyond-accuracy-card" data-testid="rec-beyond-accuracy-card" onClick={onClickArtifact}>
      <div className="rec-card-header">
        <div className="flex items-center gap-2">
          <Compass size={15} className="text-cyan-400" />
          <h4 className="rec-card-title">Beyond-Accuracy & Diversity</h4>
        </div>
        <span className="text-xs text-muted">Serendipity & Novelty</span>
      </div>

      <div className="rec-stats-grid">
        <div className="rec-stat-box">
          <div className="text-xs text-muted mb-1">Catalog Coverage</div>
          <div className="rec-stat-value text-cyan-300">{(catalogCoverage * 100).toFixed(1)}%</div>
          <div className="text-[11px] text-muted mt-0.5">Unique items recommended</div>
        </div>

        <div className="rec-stat-box">
          <div className="text-xs text-muted mb-1">User Coverage</div>
          <div className="rec-stat-value text-cyan-300">{(userCoverage * 100).toFixed(1)}%</div>
          <div className="text-[11px] text-muted mt-0.5">Users receiving valid recs</div>
        </div>

        <div className="rec-stat-box">
          <div className="text-xs text-muted mb-1">Novelty Score</div>
          <div className="rec-stat-value text-amber-300">{novelty.toFixed(2)}</div>
          <div className="text-[11px] text-muted mt-0.5">Self-information bits</div>
        </div>

        <div className="rec-stat-box">
          <div className="text-xs text-muted mb-1">Popularity Bias</div>
          <div className="rec-stat-value text-slate-200">{popularityBias.toFixed(1)}</div>
          <div className="text-[11px] text-muted mt-0.5">Avg item interaction count</div>
        </div>
      </div>
    </div>
  )
}

/**
 * 4. Cold-Start Cohort Card
 */
export const RecommenderColdStartCard: React.FC<{
  coldStartData?: any
  onClickArtifact?: () => void
}> = ({ coldStartData, onClickArtifact }) => {
  if (!coldStartData) {
    return (
      <div className="rec-card rec-coldstart-card empty-card" data-testid="rec-coldstart-card">
        <div className="rec-card-header">
          <div className="flex items-center gap-2">
            <Zap size={15} className="text-amber-400" />
            <h4 className="rec-card-title">Cold-Start Cohort Analysis</h4>
          </div>
        </div>
        <p className="text-xs text-muted">Awaiting cold-start evaluation artifact...</p>
      </div>
    )
  }

  const cohorts = coldStartData.cohorts || []
  const warmCohort = cohorts.find((c: any) => c.cohort.toLowerCase().includes('warm'))
  const coldCohort = cohorts.find((c: any) => c.cohort.toLowerCase().includes('cold'))

  const warmNdcg = warmCohort?.metrics?.['ndcg@10'] ?? warmCohort?.metrics?.ndcg ?? 0
  const coldNdcg = coldCohort?.metrics?.['ndcg@10'] ?? coldCohort?.metrics?.ndcg ?? 0
  const warmCount = warmCohort?.count ?? 0
  const coldCount = coldCohort?.count ?? 0

  const degradationRatio = coldStartData.ndcg_degradation_ratio ?? coldStartData.degradation_ratio ?? (warmNdcg > 0 ? (warmNdcg - coldNdcg) / warmNdcg : 0)
  const degradationPct = degradationRatio * 100

  return (
    <div className="rec-card rec-coldstart-card" data-testid="rec-coldstart-card" onClick={onClickArtifact}>
      <div className="rec-card-header">
        <div className="flex items-center gap-2">
          <Zap size={15} className="text-amber-400" />
          <h4 className="rec-card-title">Cold-Start Cohort Robustness</h4>
        </div>
        <div className="flex items-center gap-1.5">
          <TrendingDown size={13} className={degradationPct > 35 ? 'text-rose-400' : 'text-amber-400'} />
          <span className="text-xs font-mono">
            {degradationPct.toFixed(1)}% Degradation
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="rec-cohort-box bg-slate-900/60 p-2.5 rounded border border-slate-700/50">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-semibold text-slate-200">Warm Users ({warmCount})</span>
            <span className="text-emerald-400 text-[11px]">Established</span>
          </div>
          <div className="text-lg font-mono font-bold text-emerald-400">
            {Number(warmNdcg).toFixed(4)}
          </div>
          <span className="text-[11px] text-muted">NDCG@10 Baseline</span>
        </div>

        <div className="rec-cohort-box bg-slate-900/60 p-2.5 rounded border border-slate-700/50">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-semibold text-slate-200">Cold Users ({coldCount})</span>
            <span className="text-amber-400 text-[11px]">≤3 Interactions</span>
          </div>
          <div className="text-lg font-mono font-bold text-amber-400">
            {Number(coldNdcg).toFixed(4)}
          </div>
          <span className="text-[11px] text-muted">NDCG@10 Under Sparsity</span>
        </div>
      </div>

      <div className="text-xs text-muted">
        <span>Cold users receive popularity fallback / side-feature factorization rankings.</span>
      </div>
    </div>
  )
}

/**
 * 5. Parameter Sensitivity Table & Stability Grid
 */
export const RecommenderSensitivityGrid: React.FC<{
  sensitivityData?: any
  onClickArtifact?: () => void
}> = ({ sensitivityData, onClickArtifact }) => {
  if (!sensitivityData) {
    return (
      <div className="rec-card rec-sensitivity-card empty-card" data-testid="rec-sensitivity-card">
        <div className="rec-card-header">
          <div className="flex items-center gap-2">
            <Sliders size={15} className="text-purple-400" />
            <h4 className="rec-card-title">Parameter Sensitivity & Stability</h4>
          </div>
        </div>
        <p className="text-xs text-muted">Awaiting sensitivity evaluation artifact...</p>
      </div>
    )
  }

  const paramName = sensitivityData.parameter_name || 'latent_dim'
  const points: any[] = sensitivityData.points || []
  const isStable = sensitivityData.is_stable ?? true
  const maxDegradation = sensitivityData.max_degradation_pct ?? 0

  return (
    <div className="rec-card rec-sensitivity-card" data-testid="rec-sensitivity-card" onClick={onClickArtifact}>
      <div className="rec-card-header">
        <div className="flex items-center gap-2">
          <Sliders size={15} className="text-purple-400" />
          <h4 className="rec-card-title">Latent Factor Sensitivity ({paramName})</h4>
        </div>
        <div className="flex items-center gap-2">
          {isStable ? (
            <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium">
              <CheckCircle2 size={13} />
              <span>Stable (Δ &lt; 25%)</span>
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-rose-400 font-medium">
              <AlertTriangle size={13} />
              <span>Unstable (Δ &gt; 25%)</span>
            </span>
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="rec-table">
          <thead>
            <tr>
              <th>{paramName}</th>
              <th>NDCG@10</th>
              <th>Recall@10</th>
              <th>Δ from Baseline</th>
            </tr>
          </thead>
          <tbody>
            {points.map((pt, idx) => {
              const delta = pt.delta_from_baseline?.ndcg ?? 0
              const deltaFormatted = delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`
              const isBase = pt.parameter_value === sensitivityData.baseline_value

              return (
                <tr key={idx} className={isBase ? 'bg-indigo-950/20' : ''}>
                  <td className="font-mono text-slate-200">
                    {pt.parameter_value} {isBase && <span className="text-[10px] text-indigo-400 ml-1">(base)</span>}
                  </td>
                  <td className="font-mono text-slate-200">
                    {Number(pt.metrics?.['ndcg@10'] ?? pt.metrics?.ndcg ?? 0).toFixed(4)}
                  </td>
                  <td className="font-mono text-slate-300">
                    {Number(pt.metrics?.['recall@10'] ?? pt.metrics?.recall ?? 0).toFixed(4)}
                  </td>
                  <td>
                    <span
                      className={`font-mono text-xs px-1.5 py-0.5 rounded ${
                        delta >= 0
                          ? 'text-emerald-400 bg-emerald-950/40'
                          : Math.abs(delta) > 0.25
                          ? 'text-rose-400 bg-rose-950/40'
                          : 'text-amber-400 bg-amber-950/40'
                      }`}
                    >
                      {deltaFormatted}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/**
 * Composite Recommender Canvas View: Assembles all specialized cards into a cohesive workstation dashboard
 */
export const RecommenderCanvasViews: React.FC<RecommenderCanvasProps> = ({
  artifacts,
  selectedArtifactId,
  onSelectArtifact,
  onHighlightEvidence,
}) => {
  // Extract artifact payloads
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

  return (
    <div className="recommender-canvas-views space-y-4" data-testid="recommender-canvas-views">
      {/* Top Banner: Sparsity & Cold Start side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RecommenderSparsityCard
          profileData={sparsityData}
          onClickArtifact={() => sparsityArt && onSelectArtifact?.(sparsityArt.artifactId)}
        />
        <RecommenderColdStartCard
          coldStartData={coldStartData}
          onClickArtifact={() => coldStartArt && onSelectArtifact?.(coldStartArt.artifactId)}
        />
      </div>

      {/* Main Ranking Cutoff Table */}
      <RecommenderRankingTable
        rankingData={rankingData}
        onClickArtifact={() => rankingArt && onSelectArtifact?.(rankingArt.artifactId)}
      />

      {/* Bottom Grid: Beyond-Accuracy & Sensitivity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RecommenderBeyondAccuracyCard
          metrics={metricsList}
          onClickArtifact={() => metricsArt && onSelectArtifact?.(metricsArt.artifactId)}
        />
        <RecommenderSensitivityGrid
          sensitivityData={sensitivityData}
          onClickArtifact={() => sensitivityArt && onSelectArtifact?.(sensitivityArt.artifactId)}
        />
      </div>
    </div>
  )
}
