import { CheckCircle2, ShieldCheck, Hash, Clock, Layers, ExternalLink, AlertTriangle, HelpCircle, Edit3, RotateCcw, ArrowUpCircle } from 'lucide-react'
import type { DecisionReceipt } from '../../contracts/types'
import { formatTimeOnly } from '../../utils/formatTimestamp'

interface DecisionReceiptCardProps {
  receipt: DecisionReceipt
  onHighlightEvidence?: (evidenceId: string) => void
  onHighlightStage?: (stageId: string) => void
}

export function DecisionReceiptCard({
  receipt,
  onHighlightEvidence,
  onHighlightStage,
}: DecisionReceiptCardProps) {
  const action = (receipt.action || 'ACCEPT').toUpperCase()

  let actionColor = 'badge-accept'
  let actionIcon = <CheckCircle2 size={13} />

  if (action === 'CHALLENGE') {
    actionColor = 'badge-challenge'
    actionIcon = <AlertTriangle size={13} />
  } else if (action === 'OVERRIDE') {
    actionColor = 'badge-override'
    actionIcon = <Edit3 size={13} />
  } else if (action === 'QUESTION') {
    actionColor = 'badge-question'
    actionIcon = <HelpCircle size={13} />
  } else if (action === 'RERUN') {
    actionColor = 'badge-rerun'
    actionIcon = <RotateCcw size={13} />
  } else if (action === 'ESCALATE') {
    actionColor = 'badge-escalate'
    actionIcon = <ArrowUpCircle size={13} />
  }

  const receiptId = receipt.receipt_id || receipt.receiptId || 'REC-UNKNOWN'
  const targetStage = receipt.target_stage || receipt.targetStage
  const targetCp = receipt.target_checkpoint || receipt.targetCheckpoint
  const evidenceIds = receipt.evidence_ids || receipt.evidenceIds || []
  const decisionHash = receipt.decision_hash || receipt.decisionHash || ''

  return (
    <div className="decision-receipt-card" role="region" aria-label={`Decision Receipt ${receiptId}`}>
      <div className="receipt-header">
        <div className="receipt-action-group">
          <span className={`receipt-badge ${actionColor}`}>
            {actionIcon}
            <span>{action}</span>
          </span>
          <code className="receipt-id">{receiptId}</code>
        </div>

        <div className="receipt-immutable-badge" title="Decision receipt recorded with SHA-256 digest; evidence records remain strictly immutable">
          <ShieldCheck size={13} className="text-sage" />
          <span>Evidence Immutable</span>
        </div>
      </div>

      <div className="receipt-body">
        <p className="receipt-rationale">{receipt.rationale}</p>

        <div className="receipt-meta-grid">
          {targetStage && (
            <div className="receipt-meta-item">
              <span className="meta-label">Target Stage:</span>
              <button
                type="button"
                className="meta-link-btn"
                onClick={() => onHighlightStage?.(targetStage)}
                title={`Focus stage ${targetStage}`}
              >
                <Layers size={11} className="mr-1 inline" />
                {targetStage}
              </button>
            </div>
          )}

          {targetCp && (
            <div className="receipt-meta-item">
              <span className="meta-label">Target Checkpoint:</span>
              <code className="meta-code">{targetCp}</code>
            </div>
          )}

          <div className="receipt-meta-item">
            <span className="meta-label">Author:</span>
            <span className="meta-val">{receipt.author || 'Risk Officer'}</span>
          </div>

          <div className="receipt-meta-item">
            <span className="meta-label">Recorded:</span>
            <span className="meta-val">
              {receipt.timestamp ? formatTimeOnly(receipt.timestamp, 'Just now') : 'Just now'}
            </span>
          </div>
        </div>

        {evidenceIds.length > 0 && (
          <div className="receipt-evidence-row">
            <span className="meta-label">Referenced Evidence:</span>
            <div className="receipt-evidence-chips">
              {evidenceIds.map((eid) => (
                <button
                  key={eid}
                  type="button"
                  className="receipt-ev-chip"
                  onClick={() => onHighlightEvidence?.(eid)}
                  title={`Focus evidence ${eid}`}
                >
                  <span>{eid}</span>
                  <ExternalLink size={9} />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="receipt-footer">
        <div className="receipt-hash-line" title={`SHA256: ${decisionHash}`}>
          <Hash size={11} className="text-muted" />
          <span className="hash-label">SHA-256 Digest:</span>
          <code className="hash-val">
            {decisionHash ? `${decisionHash.slice(0, 20)}...` : '0x8f3c...'}
          </code>
        </div>
        <span className="receipt-status-pill">{receipt.status || 'RECORDED'}</span>
      </div>
    </div>
  )
}
