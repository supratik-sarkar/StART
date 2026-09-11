import { ArrowUpRight, Clock, Fingerprint, Layers, Link2, Pin, ShieldCheck } from 'lucide-react'
import type { EvidenceRecord } from '../../contracts/types'
import { formatDateTime } from '../../utils/formatTimestamp'

export function EvidenceInspector({
  evidence,
  onPinEvidence,
  isPinned,
}: {
  evidence: EvidenceRecord | null
  onPinEvidence?: (ev: EvidenceRecord) => void
  isPinned?: boolean
}) {
  if (!evidence) {
    return (
      <div className="inspector-empty">
        <ShieldCheck size={28} />
        <strong>Select runtime evidence</strong>
        <span>Inspect Evidence ID, deterministic test surface, metrics, status, provenance, producer stage, and timestamp.</span>
      </div>
    )
  }

  const formattedTime = evidence.createdAt
    ? formatDateTime(evidence.createdAt, 'Timestamp unavailable')
    : 'Recorded at run time'

  return (
    <div className="evidence-inspector">
      <div className="inspector-head">
        <div>
          <div className="evidence-id-row flex items-center gap-2">
            <span className="mono evidence-id-tag">{evidence.evidenceId}</span>
            <span className="evidence-test-tag mono">{evidence.testId}</span>
            {onPinEvidence && (
              <button
                type="button"
                className={`p-1 rounded text-xs transition-colors flex items-center gap-1 ${
                  isPinned
                    ? 'text-[#fc8181] bg-red-950/40 border border-red-800/40 font-medium'
                    : 'text-[#718096] hover:text-white bg-[#1a202c] border border-[#2d3748]'
                }`}
                onClick={() => onPinEvidence(evidence)}
                title={isPinned ? 'Unpin evidence' : 'Pin evidence to workspace'}
                data-testid={`pin-evidence-btn-${evidence.evidenceId}`}
              >
                <Pin size={11} className={isPinned ? 'fill-current' : ''} />
                <span className="text-[10px]">{isPinned ? 'Pinned' : 'Pin'}</span>
              </button>
            )}
          </div>
          <h3>{evidence.title}</h3>
          <p>{evidence.summary}</p>
        </div>
        <span className={`evidence-status ${evidence.status.toLowerCase()}`}>{evidence.status}</span>
      </div>

      <div className="evidence-field-table">
        <div className="evidence-field-row">
          <span className="field-name"><Fingerprint size={12} /> Evidence ID</span>
          <strong className="field-val mono">{evidence.evidenceId}</strong>
        </div>
        <div className="evidence-field-row">
          <span className="field-name"><Layers size={12} /> Test surface</span>
          <strong className="field-val mono">{evidence.testId}</strong>
        </div>
        <div className="evidence-field-row">
          <span className="field-name"><ShieldCheck size={12} /> Status</span>
          <strong className={`field-val status-text-${evidence.status.toLowerCase()}`}>{evidence.status}</strong>
        </div>
        <div className="evidence-field-row">
          <span className="field-name"><Layers size={12} /> Producer stage</span>
          <strong className="field-val">{evidence.parentNodeId || 'preflight'}</strong>
        </div>
        <div className="evidence-field-row">
          <span className="field-name"><Clock size={12} /> Timestamp</span>
          <strong className="field-val">{formattedTime}</strong>
        </div>
      </div>

      <div className="block-title">Deterministic Metrics ({evidence.metrics.length})</div>
      <div className="metric-grid">
        {evidence.metrics.map(m => (
          <div key={m.name} className="metric-cell">
            <span>{m.name.replaceAll('_', ' ')}</span>
            <strong>
              {String(m.value)}
              {m.unit && <small> {m.unit}</small>}
            </strong>
            {m.criterion && <em>{m.criterion}</em>}
          </div>
        ))}
      </div>

      <div className="provenance-block">
        <div className="block-title"><Link2 size={14} /> Provenance</div>
        {evidence.provenance.map(p => (
          <div key={p} className="provenance-row">
            <span className="provenance-dot" />
            <code>{p}</code>
          </div>
        ))}
      </div>
    </div>
  )
}
