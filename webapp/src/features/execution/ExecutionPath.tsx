import type { AgentPlanStep, ExecutionGraph, HandoffTransition, ProgressState } from '../../contracts/types'
import { AgentStageCard } from './AgentStageCard'

export function ExecutionPath({
  plan,
  graph,
  progress,
  selected,
  handoffs,
  onSelect,
  onHighlightEvidence,
}: {
  plan: AgentPlanStep[]
  graph: ExecutionGraph
  progress?: ProgressState
  selected: string | null
  handoffs?: HandoffTransition[]
  onSelect: (id: string) => void
  onHighlightEvidence?: (evidenceId: string) => void
}) {
  const currentRunningStep = plan.find((s) => s.status === 'running') || plan.find((s) => s.status === 'completed')

  return (
    <section className="execution-path-panel" aria-label="Deterministic Stage Chronology">
      <div className="panel-kicker">Deterministic Stage Chronology</div>

      <div className="journey-header">
        <div>
          <strong>{progress?.label || 'Preparing deterministic execution'}</strong>
          <span>{progress?.detail || 'Runtime events will update stage states.'}</span>
        </div>
        {progress?.percent != null && <b>{Math.round(progress.percent)}%</b>}
      </div>

      {progress?.percent != null && (
        <div className="run-progress">
          <div style={{ width: `${progress.percent}%` }} />
          <span className="progress-orb" style={{ left: `calc(${progress.percent}% - 6px)` }} />
        </div>
      )}

      <div className="journey-stages-container">
        {plan.map((step) => {
          const match = graph.nodes.find((n) => n.id === step.id)
          const isSelected = selected === step.id
          const isCurrent = currentRunningStep?.id === step.id

          // Check if any active handoff matches this stage
          const activeHandoff = handoffs?.find(
            (h) => h.active && (h.stage.toLowerCase() === step.label.toLowerCase() || h.action.includes(step.id))
          )

          return (
            <AgentStageCard
              key={step.id}
              step={step}
              graphNode={match}
              isSelected={isSelected}
              isCurrent={isCurrent}
              handoff={activeHandoff}
              onSelect={onSelect}
              onHighlightEvidence={onHighlightEvidence}
            />
          )
        })}
      </div>
    </section>
  )
}
