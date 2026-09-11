import { useState, useEffect } from 'react'
import { Braces, CheckCircle2, CircleDot, FlaskConical, GitBranch, Wrench, ChevronDown, ChevronRight } from 'lucide-react'
import type { RuntimeEvent } from '../../contracts/types'
import { formatTimeOnly } from '../../utils/formatTimestamp'

const icon = (t: string) =>
  t.includes('tool') ? <Wrench size={14} /> :
  t.includes('evidence') ? <Braces size={14} /> :
  t.includes('test') ? <FlaskConical size={14} /> :
  t.includes('phase') ? <GitBranch size={14} /> :
  <CircleDot size={14} />

export function EventStream({
  events,
  onSelect,
  isCompleted = false,
}: {
  events: RuntimeEvent[]
  onSelect: (id: string) => void
  isCompleted?: boolean
}) {
  const [isCollapsed, setIsCollapsed] = useState(isCompleted)

  useEffect(() => {
    if (isCompleted) {
      setIsCollapsed(true)
    }
  }, [isCompleted])

  return (
    <section className={`event-panel ${isCollapsed ? 'collapsed' : ''}`} data-testid="event-stream-panel">
      <div
        className="panel-title flex items-center justify-between cursor-pointer select-none"
        onClick={() => setIsCollapsed(!isCollapsed)}
        id="toggle-event-stream-btn"
      >
        <div className="flex items-center gap-2">
          {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
          {isCompleted ? (
            <span>Execution Trace · {events.length} events {isCollapsed ? '[collapsed]' : ''}</span>
          ) : (
            <>
              <span className="live-dot" />
              <span>Live execution</span>
            </>
          )}
        </div>
        <span className="text-xs text-muted">
          {isCollapsed ? 'Click to expand' : `${events.length} events`}
        </span>
      </div>

      {!isCollapsed && (
        <div className="event-list">
          {events.length === 0 ? (
            <div className="empty-state">Runtime events will appear here when deterministic execution starts.</div>
          ) : (
            [...events].reverse().map((e) => (
              <button
                key={e.eventId}
                onClick={() => e.nodeId && onSelect(e.nodeId)}
                className="event-row"
                type="button"
              >
                <div className="event-icon">{icon(e.type)}</div>
                <div>
                  <div className="event-top">
                    <strong>{e.title}</strong>
                    <time>{formatTimeOnly(e.timestamp, 'Timestamp unavailable')}</time>
                  </div>
                  <p>{e.message}</p>
                  {e.evidenceIds?.length ? (
                    <div className="event-tags">
                      {e.evidenceIds.map((x) => (
                        <span key={x}>{x}</span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </section>
  )
}

