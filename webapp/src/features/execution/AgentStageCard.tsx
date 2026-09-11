import { useState } from 'react'
import {
  Check,
  LoaderCircle,
  AlertCircle,
  ShieldAlert,
  Circle,
  HelpCircle,
  ArrowRight,
  Info,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import type { AgentPlanStep, ExecutionGraphNode, HandoffTransition } from '../../contracts/types'

interface AgentStageCardProps {
  step: AgentPlanStep
  graphNode?: ExecutionGraphNode
  isSelected: boolean
  isCurrent: boolean
  handoff?: HandoffTransition
  onSelect: (id: string) => void
  onHighlightEvidence?: (evidenceId: string) => void
}

const STAGE_SYMBOLS: Record<string, string> = {
  'step-context': '◈',
  'step-preflight': '◇',
  'step-features': '◎',
  'step-supervised': '△',
  'step-deep_learning': '△',
  'step-tuning': '△',
  'step-pricing': '△',
  'step-portfolio': '△',
  'step-evidence': '✦',
  'step-governance': '◆',
  'step-attestation': '⬡',
}

const STAGE_WHY_RATIONALE: Record<string, string> = {
  'step-context': 'Initializes seeded execution context, resolving shape, target distribution, and baseline metadata.',
  'step-preflight': 'Executes deterministic data contract, null rate, duplicate row, and typing assertions to fail early on corruption.',
  'step-features': 'Assesses feature multicollinearity, distribution drift, stationarity, and attribution stability before training.',
  'step-supervised': 'Evaluates model discrimination (ROC-AUC, Gini, KS) and probabilistic reliability (ECE, Brier) against holdout data.',
  'step-deep_learning': 'Inspects neural network layer spectra, activation distributions, and gradient stability.',
  'step-pricing': 'Runs interest rate term structure and CEV diffusion diagnostics.',
  'step-portfolio': 'Evaluates market risk, Kupiec VaR exception tests, and portfolio covariance matrices.',
  'step-evidence': 'Gathers all deterministic test outputs into an immutable evidence bundle registered in the canonical ledger.',
  'step-governance': 'Evaluates codified model risk policy rules against grounded evidence surfaces without human bias.',
  'step-attestation': 'Constructs and cryptographically signs the Merkle attestation seal over the complete evidence history.',
}

const STAGE_SKIPPED_INFO: Record<string, { skippedCount: number; reason: string }> = {
  'step-context': { skippedCount: 0, reason: 'All context initialization assertions are strictly required.' },
  'step-preflight': { skippedCount: 0, reason: 'Preflight data checks are fully applicable to tabular contexts.' },
  'step-features': { skippedCount: 2, reason: 'Time-series autocorrelation tests skipped: context is cross-sectional tabular benchmark.' },
  'step-supervised': { skippedCount: 1, reason: 'Survival analysis metrics skipped: context target is binary classification.' },
  'step-evidence': { skippedCount: 0, reason: 'Evidence aggregation applies to all produced test outputs.' },
  'step-governance': { skippedCount: 0, reason: 'Tier-1 materiality mandates full governance policy check.' },
}

export function AgentStageCard({
  step,
  graphNode,
  isSelected,
  isCurrent,
  handoff,
  onSelect,
  onHighlightEvidence,
}: AgentStageCardProps) {
  const [showWhy, setShowWhy] = useState(false)
  const [showSkipped, setShowSkipped] = useState(false)

  const rawStatus = (graphNode?.status || step.status || 'future').toLowerCase()
  const isCompleted = rawStatus === 'completed' || rawStatus === 'success'
  const isRunning = rawStatus === 'running'
  const isFailed = rawStatus === 'failed' || rawStatus === 'error'
  const isBlocked = rawStatus === 'blocked'

  // Determine truthful stage kind
  const stageKind = step.kind || graphNode?.kind || 'test'
  let badgeLabel = 'DETERMINISTIC'
  let badgeClass = 'badge-deterministic'

  if (stageKind === 'governance') {
    badgeLabel = 'POLICY'
    badgeClass = 'badge-policy'
  } else if (stageKind === 'human') {
    badgeLabel = 'HUMAN'
    badgeClass = 'badge-human'
  } else if (stageKind === 'agent') {
    badgeLabel = 'AGENT'
    badgeClass = 'badge-agent'
  } else {
    // Pure calculation / test engine / context / evidence
    badgeLabel = 'DETERMINISTIC'
    badgeClass = 'badge-deterministic'
  }

  const symbol = STAGE_SYMBOLS[step.id] || '◉'
  const whyRationale = STAGE_WHY_RATIONALE[step.id] || 'Standard deterministic review step mandated by protocol.'
  const skippedInfo = STAGE_SKIPPED_INFO[step.id]

  return (
    <div
      className={`agent-stage-card ${isSelected ? 'is-selected' : ''} ${isCurrent ? 'is-current' : ''} status-${rawStatus}`}
      onClick={() => onSelect(step.id)}
    >
      <div className="stage-card-header">
        <div className="stage-symbol-box" title={`Stage symbol: ${symbol}`}>
          <span className="stage-symbol">{symbol}</span>
        </div>

        <div className="stage-info-column">
          <div className="stage-title-line">
            <span className="stage-title">{step.label}</span>
            <span className={`stage-type-badge ${badgeClass}`}>{badgeLabel}</span>
          </div>
          <div className="stage-desc">{graphNode?.subtitle || step.description || step.id}</div>
        </div>

        <div className="stage-status-indicator">
          {isCompleted ? (
            <span className="status-icon icon-completed" title="Completed">
              <Check size={14} />
            </span>
          ) : isRunning ? (
            <span className="status-icon icon-running" title="Running">
              <LoaderCircle className="spin" size={14} />
            </span>
          ) : isFailed ? (
            <span className="status-icon icon-failed" title="Failed">
              <AlertCircle size={14} />
            </span>
          ) : isBlocked ? (
            <span className="status-icon icon-blocked" title="Blocked">
              <ShieldAlert size={14} />
            </span>
          ) : (
            <span className="status-icon icon-future" title="Planned">
              <Circle size={10} />
            </span>
          )}
        </div>
      </div>

      {/* Real Agent Handoff Indicator (150-250ms restrained indigo pulse) */}
      {handoff && handoff.active && (
        <div className="stage-handoff-banner" role="status" aria-label="Agent Handoff">
          <div className="handoff-pulse-dot" />
          <span className="handoff-text">
            Handoff: <strong>{handoff.sourceAgent}</strong>
            <ArrowRight size={11} className="handoff-arrow" />
            <strong>{handoff.targetAgent}</strong>
          </span>
          <span className="handoff-action">{handoff.action}</span>
        </div>
      )}

      {/* Produced Evidence References */}
      {graphNode?.evidenceIds && graphNode.evidenceIds.length > 0 && (
        <div className="stage-evidence-chips">
          <span className="chips-label">Stage Refs ({graphNode.evidenceIds.length}):</span>
          {graphNode.evidenceIds.slice(0, 4).map((eid) => (
            <button
              key={eid}
              type="button"
              className="evidence-ref-chip"
              onClick={(e) => {
                e.stopPropagation()
                onHighlightEvidence?.(eid)
              }}
              title={`Inspect Evidence ${eid}`}
            >
              {eid}
            </button>
          ))}
          {graphNode.evidenceIds.length > 4 && (
            <span className="evidence-overflow-count">+{graphNode.evidenceIds.length - 4} more</span>
          )}
        </div>
      )}

      {/* Inspectors: Why this stage? & Why skipped? */}
      <div className="stage-card-footer" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="stage-inspector-btn"
          onClick={() => setShowWhy((v) => !v)}
          title="Why was this stage included?"
        >
          <Info size={11} />
          <span>Why this stage?</span>
          {showWhy ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
        </button>

        {skippedInfo && (
          <button
            type="button"
            className="stage-inspector-btn"
            onClick={() => setShowSkipped((v) => !v)}
            title="Inspect skipped tests"
          >
            <HelpCircle size={11} />
            <span>Why skipped? {skippedInfo.skippedCount > 0 ? `(${skippedInfo.skippedCount})` : ''}</span>
            {showSkipped ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </button>
        )}
      </div>

      {showWhy && (
        <div className="inspector-popover why-popover" onClick={(e) => e.stopPropagation()}>
          <div className="inspector-title">Protocol Selection Rationale</div>
          <div className="inspector-body">{whyRationale}</div>
        </div>
      )}

      {showSkipped && skippedInfo && (
        <div className="inspector-popover skipped-popover" onClick={(e) => e.stopPropagation()}>
          <div className="inspector-title">Applicability Resolution</div>
          <div className="inspector-body">
            <strong>{skippedInfo.skippedCount} candidate test(s) skipped:</strong> {skippedInfo.reason}
          </div>
        </div>
      )}
    </div>
  )
}
