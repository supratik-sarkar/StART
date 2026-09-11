import { useState } from 'react'
import {
  Check,
  Shield,
  ShieldAlert,
  LoaderCircle,
  HelpCircle,
  X,
  ExternalLink,
  Hash,
  Clock,
  UserCheck,
  Layers,
} from 'lucide-react'
import type { CheckpointRecord } from '../../contracts/types'
import { formatTimeOnly } from '../../utils/formatTimestamp'

interface CheckpointRailProps {
  checkpoints: CheckpointRecord[]
  selectedCheckpointId?: string | null
  onSelectCheckpoint?: (cp: CheckpointRecord) => void
  onHighlightEvidence?: (evidenceId: string) => void
}

const CANONICAL_CHECKPOINT_DEFS = [
  { id: 'CP-001', name: 'Context ready', producingStage: 'step-context', agent: 'Director' },
  { id: 'CP-002', name: 'Preflight complete', producingStage: 'step-preflight', agent: 'DataSpecialist' },
  { id: 'CP-003', name: 'Feature analysis', producingStage: 'step-features', agent: 'FeatureSpecialist' },
  { id: 'CP-004', name: 'Model evaluation', producingStage: 'step-supervised', agent: 'ModelSpecialist' },
  { id: 'CP-005', name: 'Evidence committed', producingStage: 'step-evidence', agent: 'EvidenceLedger' },
  { id: 'CP-006', name: 'Attestation signed', producingStage: 'step-governance', agent: 'ModelGovernance' },
]

export function CheckpointRail({
  checkpoints,
  selectedCheckpointId,
  onSelectCheckpoint,
  onHighlightEvidence,
}: CheckpointRailProps) {
  const [activeModalCp, setActiveModalCp] = useState<CheckpointRecord | null>(null)

  // Map known checkpoints
  const cpMap = new Map<string, CheckpointRecord>()
  checkpoints.forEach((c) => cpMap.set(c.checkpoint_id || (c as any).checkpointId, c))

  const handleNodeClick = (defId: string) => {
    const existing = cpMap.get(defId)
    if (existing) {
      setActiveModalCp(existing)
      onSelectCheckpoint?.(existing)
    } else {
      // Future or pending checkpoint
      const fallback: CheckpointRecord = {
        checkpoint_id: defId,
        name: CANONICAL_CHECKPOINT_DEFS.find((d) => d.id === defId)?.name || defId,
        status: 'pending',
        producing_stage: CANONICAL_CHECKPOINT_DEFS.find((d) => d.id === defId)?.producingStage || '',
        agent_signature: CANONICAL_CHECKPOINT_DEFS.find((d) => d.id === defId)?.agent || '',
        evidence_ids: [],
        commit_hash: '',
        timestamp: '',
        summary: 'Checkpoint scheduled in canonical execution sequence.',
      }
      setActiveModalCp(fallback)
    }
  }

  return (
    <section className="checkpoint-rail-section" aria-label="Verification Milestone Rail">
      <div className="checkpoint-rail-header">
        <div className="rail-title-group">
          <Shield size={14} className="rail-icon" />
          <span className="rail-title">Verification Milestone Rail</span>
        </div>
        <span className="rail-counter">
          {checkpoints.filter((c) => c.status === 'completed').length} of {CANONICAL_CHECKPOINT_DEFS.length} Verified
        </span>
      </div>

      <div className="checkpoint-track">
        {CANONICAL_CHECKPOINT_DEFS.map((def, idx) => {
          const rec = cpMap.get(def.id)
          const status = rec?.status || 'pending'
          const isCompleted = status === 'completed'
          const isActive = status === 'active' || status === 'running'
          const isBlocked = status === 'blocked'
          const isHumanPending = status === 'human-pending'

          let statusClass = 'cp-pending'
          if (isCompleted) statusClass = 'cp-completed'
          else if (isActive) statusClass = 'cp-active'
          else if (isBlocked) statusClass = 'cp-blocked'
          else if (isHumanPending) statusClass = 'cp-human-pending'

          const isSelected = selectedCheckpointId === def.id || activeModalCp?.checkpoint_id === def.id

          return (
            <div key={def.id} className="checkpoint-node-wrapper">
              <button
                type="button"
                className={`checkpoint-node ${statusClass} ${isSelected ? 'is-selected' : ''}`}
                onClick={() => handleNodeClick(def.id)}
                title={`Milestone ${def.id}: ${def.name} (${status.toUpperCase()})`}
                aria-label={`Inspect milestone ${def.id}: ${def.name}`}
              >
                <div className="cp-circle">
                  {isCompleted ? (
                    <Check size={12} className="cp-check" />
                  ) : isActive ? (
                    <LoaderCircle size={12} className="spin cp-spin" />
                  ) : isBlocked ? (
                    <ShieldAlert size={12} className="cp-alert" />
                  ) : isHumanPending ? (
                    <HelpCircle size={12} className="cp-human" />
                  ) : (
                    <span className="cp-index">{idx + 1}</span>
                  )}
                </div>
                <span className="cp-label-id">{def.id}</span>
              </button>

              {idx < CANONICAL_CHECKPOINT_DEFS.length - 1 && (
                <div className={`cp-connector ${isCompleted ? 'connector-done' : ''}`} />
              )}
            </div>
          )
        })}
      </div>

      {/* Checkpoint Inspector Modal */}
      {activeModalCp && (
        <div className="modal-backdrop" onClick={() => setActiveModalCp(null)}>
          <div
            className="checkpoint-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-cp-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="checkpoint-modal-header">
              <div className="modal-title-line">
                <Shield size={18} className="modal-shield-icon" />
                <div>
                  <h3 id="modal-cp-title" className="modal-cp-id">
                    Milestone {activeModalCp.checkpoint_id}: {activeModalCp.name}
                  </h3>
                  <span className={`modal-status-badge status-${activeModalCp.status}`}>
                    {activeModalCp.status.toUpperCase()}
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setActiveModalCp(null)}
                aria-label="Close modal"
              >
                <X size={16} />
              </button>
            </div>

            <div className="checkpoint-modal-body">
              <div className="modal-field-row">
                <span className="field-label">Summary:</span>
                <p className="field-value summary-value">{activeModalCp.summary}</p>
              </div>

              <div className="modal-grid-two">
                <div className="modal-field-item">
                  <span className="field-label">
                    <Layers size={12} className="mr-1 inline" /> Producing Stage:
                  </span>
                  <code className="field-code">{activeModalCp.producing_stage || 'N/A'}</code>
                </div>

                <div className="modal-field-item">
                  <span className="field-label">
                    <UserCheck size={12} className="mr-1 inline" /> Agent Signature:
                  </span>
                  <strong className="field-agent">{activeModalCp.agent_signature || 'DeterministicEngine'}</strong>
                </div>

                <div className="modal-field-item">
                  <span className="field-label">
                    <Hash size={12} className="mr-1 inline" /> Commit Hash / Root:
                  </span>
                  <code className="field-hash" title={activeModalCp.commit_hash}>
                    {activeModalCp.commit_hash ? `${activeModalCp.commit_hash.slice(0, 16)}...` : 'Pending'}
                  </code>
                </div>

                <div className="modal-field-item">
                  <span className="field-label">
                    <Clock size={12} className="mr-1 inline" /> Observed At:
                  </span>
                  <span className="field-timestamp">
                    {activeModalCp.timestamp
                      ? formatTimeOnly(activeModalCp.timestamp, 'Pending')
                      : 'Pending'}
                  </span>
                </div>
              </div>

              {/* Evidence Snapshot */}
              <div className="modal-evidence-snapshot">
                <span className="snapshot-title">
                  Evidence Snapshot ({activeModalCp.evidence_ids?.length || 0} Records):
                </span>
                {activeModalCp.evidence_ids && activeModalCp.evidence_ids.length > 0 ? (
                  <div className="modal-evidence-list">
                    {activeModalCp.evidence_ids.map((eid) => (
                      <button
                        key={eid}
                        type="button"
                        className="modal-evidence-chip"
                        onClick={() => {
                          onHighlightEvidence?.(eid)
                          setActiveModalCp(null)
                        }}
                        title={`Focus evidence ${eid}`}
                      >
                        <span>{eid}</span>
                        <ExternalLink size={10} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <span className="text-muted text-xs">No direct evidence bound to context initialization milestone.</span>
                )}
              </div>
            </div>

            <div className="checkpoint-modal-footer">
              <span className="modal-footer-note">
                Verification milestone observed in execution sequence.
              </span>
              <button
                type="button"
                className="btn-modal-done"
                onClick={() => setActiveModalCp(null)}
              >
                Close Milestone Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
