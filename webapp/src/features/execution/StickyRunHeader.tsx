import { useState } from 'react'
import {
  CheckCircle2,
  Clock,
  Copy,
  Check,
  Shield,
  Layers,
  FileCode,
  FileCheck,
  MessageSquare,
  AlertTriangle,
  ExternalLink,
  ChevronLeft,
  Columns,
  GitFork,
  Pin,
  Search,
} from 'lucide-react'
import { Brand } from '../../components/Brand'
import type { CheckpointRecord, DecisionReceipt, RunSnapshot } from '../../contracts/types'

interface StickyRunHeaderProps {
  run: RunSnapshot
  checkpoints: CheckpointRecord[]
  decisions: DecisionReceipt[]
  evidenceCount: number
  artifactsCount: number
  isReplay?: boolean
  pinnedCount?: number
  onOpenThread: () => void
  onOpenChallenge: () => void
  onOpenHistory?: () => void
  onOpenCompare?: () => void
  onOpenSearch?: () => void
  onOpenPins?: () => void
  onLoadRun?: (runId: string) => void
  onBack?: () => void
}

export function StickyRunHeader({
  run,
  checkpoints,
  decisions,
  evidenceCount,
  artifactsCount,
  isReplay,
  pinnedCount,
  onOpenThread,
  onOpenChallenge,
  onOpenHistory,
  onOpenCompare,
  onOpenSearch,
  onOpenPins,
  onLoadRun,
  onBack,
}: StickyRunHeaderProps) {
  const [copied, setCopied] = useState(false)
  const [showGoalTooltip, setShowGoalTooltip] = useState(false)

  const copyRunId = () => {
    navigator.clipboard.writeText(run.runId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const completedCheckpoints = checkpoints.filter((c) => c.status === 'completed').length
  const totalCheckpoints = Math.max(6, checkpoints.length)

  const isCompleted = run.phase === 'completed'
  // Determine stage & agent
  const currentStage = run.plan.find((s) => s.status === 'running') ||
    (isCompleted ? run.plan[run.plan.length - 1] : run.plan.find((s) => s.status === 'completed')) ||
    run.plan[0]
  const isDeterministic = currentStage?.kind === 'test' || currentStage?.kind === 'context' || currentStage?.kind === 'evidence'

  return (
    <header className="sticky-run-header" role="banner" aria-label="Active Run Header">
      <div className="sticky-run-left">
        {onBack && (
          <>
            <button className="back-btn" onClick={onBack} title="Return to composer">
              <ChevronLeft size={16} />
            </button>
            <Brand />
            <span className="header-divider" />
          </>
        )}
        <div className="run-id-badge" onClick={copyRunId} title="Click to copy Run ID">
          <span className="run-id-label">{run.runId}</span>
          <button type="button" className="copy-btn" aria-label="Copy run id">
            {copied ? <Check size={13} className="text-sage" /> : <Copy size={13} />}
          </button>
        </div>

        {isReplay && (
          <div
            className="replay-badge"
            data-testid="replay-badge"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              fontWeight: 700,
              letterSpacing: '0.06em',
              background: 'rgba(217, 119, 6, 0.2)',
              color: '#fcd34d',
              border: '1px solid rgba(245, 158, 11, 0.4)',
              textTransform: 'uppercase',
            }}
          >
            <Clock size={10} />
            <span>REPLAY / READ-ONLY</span>
          </div>
        )}

        {run.parentRunId && (
          <button
            type="button"
            className="parent-lineage-btn parent-lineage-link"
            data-testid="parent-lineage-pill"
            onClick={() => (onLoadRun ? onLoadRun(run.parentRunId!) : onOpenCompare?.())}
            title={`View parent run: ${run.parentRunId}`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'monospace',
              background: '#1a2332',
              color: '#63e6be',
              border: '1px solid #2b3a50',
              cursor: 'pointer',
            }}
          >
            <GitFork size={11} />
            <span>Parent: {run.parentRunId.slice(0, 10)}…</span>
          </button>
        )}

        <div className="header-divider" />

        <div className="header-meta-group">
          <span className="header-meta-item">
            <Layers size={13} className="meta-icon" />
            <span className="meta-key">Workflow:</span>
            <strong className="meta-val">{run.workflowId}</strong>
          </span>

          <span className="header-meta-item">
            <FileCode size={13} className="meta-icon" />
            <span className="meta-key">Context:</span>
            <span className="meta-val context-val">{run.contextId}</span>
          </span>

          <div
            className="header-meta-item goal-item"
            onMouseEnter={() => setShowGoalTooltip(true)}
            onMouseLeave={() => setShowGoalTooltip(false)}
          >
            <span className="meta-key">Goal:</span>
            <span className="meta-val goal-val">{run.goal}</span>
            {showGoalTooltip && (
              <div className="goal-tooltip-popover" role="tooltip">
                <div className="goal-tooltip-title">Run Objective & Scope</div>
                <div className="goal-tooltip-content">{run.goal}</div>
                {run.parentRunId && (
                  <div className="goal-tooltip-lineage">
                    <span>Parent Run: {run.parentRunId}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="sticky-run-right">
        {/* Stage & Agent Badge */}
        <div className="stage-agent-pill">
          <span className={`pill-badge ${isCompleted ? 'badge-completed' : isDeterministic ? 'badge-deterministic' : 'badge-agent'}`}>
            {isCompleted ? 'VERIFIED' : isDeterministic ? 'DETERMINISTIC' : 'AGENT'}
          </span>
          <span className="pill-stage-name">{isCompleted ? '✓ All Stages Completed' : (currentStage?.label || 'Execution')}</span>
        </div>

        {/* Checkpoint Counter */}
        <div className="header-stat-chip" title="Verified Checkpoints">
          <Shield size={13} className="stat-icon" />
          <span>
            {completedCheckpoints}/{totalCheckpoints} CP
          </span>
        </div>

        {/* Evidence Counter */}
        <div className="header-stat-chip" title="Committed Evidence Records">
          <FileCheck size={13} className="stat-icon" />
          <span>{evidenceCount} Evidence</span>
        </div>

        {/* Artifacts Counter */}
        <div className="header-stat-chip" title="Live Review Artifacts">
          <Layers size={13} className="stat-icon" />
          <span>{artifactsCount} Artifacts</span>
        </div>

        {/* Navigation Toolbar Actions */}
        <div className="header-nav-actions flex items-center gap-1.5 mr-2">
          {onOpenSearch && (
            <button
              type="button"
              id="open-search-topbar-btn"
              className="btn-header-action"
              onClick={onOpenSearch}
              title="Global Command Palette (Cmd+K)"
              data-testid="header-search-btn"
            >
              <Search size={12} />
              <span>Search ⌘K</span>
            </button>
          )}

          {onOpenHistory && (
            <button
              type="button"
              id="open-history-topbar-btn"
              className="btn-header-action"
              onClick={onOpenHistory}
              title="Run History Ledger"
              data-testid="header-history-btn"
            >
              <Clock size={12} />
              <span>History</span>
            </button>
          )}

          {onOpenCompare && (
            <button
              type="button"
              id="open-compare-topbar-btn"
              className="btn-header-action"
              onClick={onOpenCompare}
              title="Deterministic Run Comparison"
              data-testid="header-compare-btn"
            >
              <Columns size={12} />
              <span>Compare</span>
            </button>
          )}

          {onOpenPins && (
            <button
              type="button"
              id="open-pins-topbar-btn"
              className="btn-header-action"
              onClick={onOpenPins}
              title="Pinned Items"
              data-testid="header-pins-btn"
            >
              <Pin size={12} />
              <span>Pins {pinnedCount != null && pinnedCount > 0 ? `(${pinnedCount})` : ''}</span>
            </button>
          )}
        </div>

        {/* Action Controls */}
        <div className="header-action-controls">
          <button
            type="button"
            className="btn-header-action btn-question"
            onClick={onOpenThread}
            title="Ask evidence-grounded question"
          >
            <MessageSquare size={13} />
            <span>? Question</span>
          </button>
          <button
            type="button"
            className="btn-header-action btn-challenge"
            onClick={onOpenChallenge}
            title="Challenge assumption or evidence"
          >
            <AlertTriangle size={13} />
            <span>⚔ Challenge</span>
          </button>
        </div>
      </div>
    </header>
  )
}
