import { Clock, Columns2, Cpu, Pin, Search, ShieldCheck, SlidersHorizontal } from 'lucide-react'
import type { ExecutionMode } from '../contracts/types'

export interface WorkbenchHeaderProps {
  mode: 'DEFINE' | 'EXECUTE' | 'INVESTIGATE'
  context?: string
  isReplay?: boolean
  runId?: string
  executionMode?: ExecutionMode
  onExecutionModeChange?: (m: ExecutionMode) => void
  profileMode?: 'workbench' | 'enterprise'
  onProfileModeToggle?: () => void
  onNew?: () => void
  onHistory?: () => void
  onCompare?: () => void
  onSearch?: () => void
  onPins?: () => void
  onRuntime?: () => void
  runtime?: string
  aiModel?: string
  aiProvider?: string
}

export function WorkbenchHeader({
  mode,
  context,
  isReplay = false,
  runId,
  executionMode = 'hybrid_workbench',
  onExecutionModeChange,
  profileMode = 'workbench',
  onProfileModeToggle,
  onNew,
  onHistory,
  onCompare,
  onSearch,
  onPins,
  onRuntime,
  runtime = 'Unverified',
  aiModel = 'Unverified',
  aiProvider = 'Unverified',
}: WorkbenchHeaderProps) {
  const modeLabels: Record<ExecutionMode, string> = {
    hybrid_workbench: 'HYBRID WORKBENCH',
    agentic_session: 'AGENTIC SESSION',
    deterministic_run: 'DETERMINISTIC RUN',
  }

  const cycleMode = () => {
    if (!onExecutionModeChange) return
    if (executionMode === 'hybrid_workbench') onExecutionModeChange('agentic_session')
    else if (executionMode === 'agentic_session') onExecutionModeChange('deterministic_run')
    else onExecutionModeChange('hybrid_workbench')
  }

  const defaultContext = profileMode === 'enterprise' ? 'Model validation workspace' : 'AI engineering workspace'

  return (
    <header className="workstation-header">
      <button className="workstation-brand" onClick={onNew} title="New workspace" aria-label="StART · New workspace">
        <span className="brand-symbol">S</span>
        <strong>
          StART<span>2.0</span>
        </strong>
      </button>
      <div className="header-divider" />
      <nav className="mode-indicator" aria-label="Workbench phase">
        {['DEFINE', 'EXECUTE', 'INVESTIGATE'].map((m) => (
          <span key={m} aria-current={mode === m ? 'step' : undefined}>
            {m}
          </span>
        ))}
      </nav>
      <div className="header-context" title={runId}>
        {isReplay ? 'REPLAY · ' : ''}
        {context || defaultContext}
      </div>
      <div className="header-actions">
        {(onExecutionModeChange || runId) && (
          <button
            className="runtime-button execution-mode-pill"
            title="Toggle execution mode: Hybrid Workbench, Agentic Session, or Deterministic Run"
            onClick={cycleMode}
            disabled={!onExecutionModeChange}
            aria-label={`Execution Mode: ${modeLabels[executionMode]}`}
          >
            <Cpu size={13} />
            <span>{modeLabels[executionMode]}</span>
          </button>
        )}
        {onProfileModeToggle && (
          <button
            className="icon-button"
            title={`Terminology Profile: ${profileMode === 'workbench' ? 'Workbench (Engineering)' : 'Enterprise (MRM/SR 11-7)'}`}
            aria-label="Toggle terminology profile"
            onClick={onProfileModeToggle}
          >
            <ShieldCheck size={16} />
          </button>
        )}
        {onSearch && (
          <button className="header-search" onClick={onSearch} aria-label="Search workspace" title="Search · Cmd+K">
            <Search size={15} />
            <span>Search</span>
            <kbd>⌘ K</kbd>
          </button>
        )}
        {onHistory && (
          <button className="icon-button" title="Run history" aria-label="Run history" onClick={onHistory}>
            <Clock size={17} />
          </button>
        )}
        {onCompare && (
          <button className="icon-button" title="Compare runs" aria-label="Compare runs" onClick={onCompare}>
            <Columns2 size={17} />
          </button>
        )}
        {onPins && (
          <button className="icon-button" title="Pinned records" aria-label="Pinned records" onClick={onPins}>
            <Pin size={16} />
          </button>
        )}
        {onRuntime && (
          <button className="runtime-button" title={`AI Backend: ${aiProvider==='openai'?'OpenAI':aiProvider} ${aiModel} · ${runtime}`} onClick={onRuntime}>
            <SlidersHorizontal size={14} />
            <span>AI · {aiProvider==='openai'?'OpenAI':aiProvider} · {aiModel} · {runtime}</span>
          </button>
        )}
      </div>
    </header>
  )
}
