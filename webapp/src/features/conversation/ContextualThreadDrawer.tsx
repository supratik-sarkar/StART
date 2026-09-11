import { useState, useEffect } from 'react'
import {
  X,
  MessageSquare,
  Send,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ExternalLink,
  Shield,
  Layers,
  FileCheck,
  Edit3,
  RotateCcw,
  ArrowUpCircle,
  LoaderCircle,
} from 'lucide-react'
import type { DecisionReceipt, EvidenceRecord, QuestionCitation, QuestionResponse } from '../../contracts/types'
import { DecisionReceiptCard } from '../execution/DecisionReceiptCard'
import { formatTimeOnly } from '../../utils/formatTimestamp'

interface ContextualThreadDrawerProps {
  isOpen: boolean
  onClose: () => void
  evidenceList: EvidenceRecord[]
  selectedEvidenceId?: string | null
  selectedStageId?: string | null
  decisions: DecisionReceipt[]
  initialTab?: 'question' | 'challenge' | 'receipts'
  onAskQuestion: (question: string, targetStage?: string, evidenceId?: string) => Promise<QuestionResponse | null>
  onRecordDecision: (
    action: 'ACCEPT' | 'QUESTION' | 'CHALLENGE' | 'OVERRIDE' | 'RERUN' | 'ESCALATE',
    rationale: string,
    targetStage?: string,
    targetCheckpoint?: string,
    evidenceIds?: string[]
  ) => Promise<DecisionReceipt | null>
  onHighlightEvidence?: (evidenceId: string) => void
  onHighlightStage?: (stageId: string) => void
}

interface ThreadMessage {
  id: string
  role: 'user' | 'assistant' | 'decision'
  text: string
  citations?: QuestionCitation[]
  receipt?: DecisionReceipt
  timestamp: string
}

export function ContextualThreadDrawer({
  isOpen,
  onClose,
  evidenceList,
  selectedEvidenceId,
  selectedStageId,
  decisions,
  initialTab = 'question',
  onAskQuestion,
  onRecordDecision,
  onHighlightEvidence,
  onHighlightStage,
}: ContextualThreadDrawerProps) {
  const [activeTab, setActiveTab] = useState<'question' | 'challenge' | 'receipts'>(initialTab)
  const [inputQuery, setInputQuery] = useState('')
  const [challengeRationale, setChallengeRationale] = useState('')
  const [selectedEvId, setSelectedEvId] = useState<string>(selectedEvidenceId || '')
  const [selectedStage, setSelectedStage] = useState<string>(selectedStageId || '')
  const [busy, setBusy] = useState(false)
  const [thread, setThread] = useState<ThreadMessage[]>([])

  useEffect(() => {
    if (initialTab) {
      setActiveTab(initialTab)
    }
  }, [initialTab, isOpen])

  // Escape key listener
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  const handleSendQuestion = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputQuery.trim() || busy) return

    const query = inputQuery.trim()
    setInputQuery('')
    setBusy(true)

    const userMsg: ThreadMessage = {
      id: `usr-${Date.now()}`,
      role: 'user',
      text: query,
      timestamp: new Date().toISOString(),
    }
    setThread((prev) => [...prev, userMsg])

    try {
      const resp = await onAskQuestion(query, selectedStage || undefined, selectedEvId || undefined)
      if (resp) {
        const botMsg: ThreadMessage = {
          id: `bot-${Date.now()}`,
          role: 'assistant',
          text: resp.answer,
          citations: resp.citations,
          receipt: resp.receipt,
          timestamp: resp.timestamp || new Date().toISOString(),
        }
        setThread((prev) => [...prev, botMsg])
      }
    } catch (err) {
      const errMsg: ThreadMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        text: `Error processing evidence query: ${(err as Error).message}`,
        timestamp: new Date().toISOString(),
      }
      setThread((prev) => [...prev, errMsg])
    } finally {
      setBusy(false)
    }
  }

  const handleSendAction = async (action: 'ACCEPT' | 'CHALLENGE' | 'OVERRIDE' | 'RERUN' | 'ESCALATE') => {
    const rationale = challengeRationale.trim() || `Human action ${action} recorded by Risk Officer.`
    setChallengeRationale('')
    setBusy(true)

    try {
      const evIds = selectedEvId ? [selectedEvId] : []
      const receipt = await onRecordDecision(action, rationale, selectedStage || undefined, undefined, evIds)
      if (receipt) {
        const decMsg: ThreadMessage = {
          id: `dec-${Date.now()}`,
          role: 'decision',
          text: rationale,
          receipt,
          timestamp: new Date().toISOString(),
        }
        setThread((prev) => [...prev, decMsg])
      }
    } catch (err) {
      // Error handled in hook
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="thread-drawer-backdrop" onClick={onClose}>
      <div
        className="contextual-thread-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Human Control & Evidence Thread"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drawer Header */}
        <div className="drawer-header">
          <div className="drawer-title-line">
            <Shield size={16} className="text-indigo mr-2" />
            <h3>Human Control & Evidence Inquiry</h3>
          </div>
          <button id="close-thread-drawer-btn" type="button" className="drawer-close-btn" onClick={onClose} aria-label="Close drawer">
            <X size={16} />
          </button>
        </div>

        {/* Action / Mode Tabs */}
        <div className="drawer-tabs-bar">
          <button
            type="button"
            className={`drawer-tab ${activeTab === 'question' ? 'active' : ''}`}
            onClick={() => setActiveTab('question')}
          >
            <HelpCircle size={13} />
            <span>? Ask Evidence</span>
          </button>
          <button
            type="button"
            className={`drawer-tab ${activeTab === 'challenge' ? 'active' : ''}`}
            onClick={() => setActiveTab('challenge')}
          >
            <AlertTriangle size={13} />
            <span>⚔ Challenge / Actions</span>
          </button>
          <button
            type="button"
            className={`drawer-tab ${activeTab === 'receipts' ? 'active' : ''}`}
            onClick={() => setActiveTab('receipts')}
          >
            <FileCheck size={13} />
            <span>Decision Receipts ({decisions.length})</span>
          </button>
        </div>

        {/* Target Context Selectors */}
        <div className="drawer-target-selectors">
          <div className="target-field">
            <label htmlFor="target-evidence-select">Target Evidence Record:</label>
            <select
              id="target-evidence-select"
              value={selectedEvId}
              onChange={(e) => setSelectedEvId(e.target.value)}
              className="target-select"
            >
              <option value="">All Evidence Surfaces</option>
              {evidenceList.map((e) => (
                <option key={e.evidenceId} value={e.evidenceId}>
                  {e.evidenceId} ({e.testId} - {e.status})
                </option>
              ))}
            </select>
          </div>

          <div className="target-field">
            <label htmlFor="target-stage-select">Target Stage:</label>
            <select
              id="target-stage-select"
              value={selectedStage}
              onChange={(e) => setSelectedStage(e.target.value)}
              className="target-select"
            >
              <option value="">All Stages</option>
              <option value="step-context">step-context</option>
              <option value="step-preflight">step-preflight</option>
              <option value="step-features">step-features</option>
              <option value="step-supervised">step-supervised</option>
              <option value="step-evidence">step-evidence</option>
              <option value="step-governance">step-governance</option>
            </select>
          </div>
        </div>

        {/* Drawer Body */}
        <div className="drawer-body">
          {activeTab === 'receipts' ? (
            <div className="receipts-list-view">
              {decisions.length === 0 ? (
                <div className="empty-receipts text-center py-6 text-muted text-sm">
                  No human action receipts recorded yet. Use the controls to record Accept, Question, or Challenge actions.
                </div>
              ) : (
                decisions.map((r) => (
                  <DecisionReceiptCard
                    key={r.receipt_id || (r as any).receiptId}
                    receipt={r}
                    onHighlightEvidence={onHighlightEvidence}
                    onHighlightStage={onHighlightStage}
                  />
                ))
              )}
            </div>
          ) : (
            <div className="thread-messages-view">
              {thread.length === 0 && (
                <div className="thread-empty-prompt">
                  <MessageSquare size={24} className="text-muted mb-2 inline" />
                  <p className="text-sm text-secondary">
                    Ask questions grounded in real EvidenceRecords. Every response includes verified citation back-links.
                  </p>
                </div>
              )}

              {thread.map((msg) => (
                <div key={msg.id} className={`thread-message message-${msg.role}`}>
                  <div className="message-meta">
                    <span className="message-author">{msg.role === 'user' ? 'Risk Officer' : 'Grounded Evidence Reviewer'}</span>
                    <span className="message-time">{formatTimeOnly(msg.timestamp, 'Timestamp unavailable')}</span>
                  </div>

                  <div className="message-content">{msg.text}</div>

                  {/* Verified Citations */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="message-citations-block">
                      <div className="citations-header">Verified Evidence Citations:</div>
                      <div className="citations-list">
                        {msg.citations.map((c) => (
                          <div key={c.evidence_id || (c as any).evidenceId} className="citation-card">
                            <button
                              type="button"
                              className="citation-id-btn"
                              onClick={() => onHighlightEvidence?.(c.evidence_id || (c as any).evidenceId)}
                              title="Focus Evidence Record"
                            >
                              <span>{c.evidence_id || (c as any).evidenceId}</span>
                              <ExternalLink size={10} />
                            </button>
                            <span className={`citation-status status-${(c.status || '').toLowerCase()}`}>
                              {c.status}
                            </span>
                            <span className="citation-snippet">{c.snippet}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Bound Decision Receipt */}
                  {msg.receipt && (
                    <div className="message-receipt-container">
                      <DecisionReceiptCard
                        receipt={msg.receipt}
                        onHighlightEvidence={onHighlightEvidence}
                        onHighlightStage={onHighlightStage}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Drawer Footer Controls */}
        <div className="drawer-footer">
          {activeTab === 'question' ? (
            <form onSubmit={handleSendQuestion} className="question-form">
              <input
                type="text"
                className="question-input"
                placeholder="Ask about evidence, thresholds, or test findings..."
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                disabled={busy}
              />
              <button type="submit" className="btn-send-question" disabled={busy || !inputQuery.trim()}>
                {busy ? <LoaderCircle size={14} className="spin" /> : <Send size={14} />}
                <span>Ask</span>
              </button>
            </form>
          ) : activeTab === 'challenge' ? (
            <div className="challenge-controls-panel">
              <textarea
                className="challenge-textarea"
                rows={2}
                placeholder="Specify rationale for challenge, override, or rerun..."
                value={challengeRationale}
                onChange={(e) => setChallengeRationale(e.target.value)}
                disabled={busy}
              />

              <div className="action-buttons-grid">
                <button
                  type="button"
                  className="btn-human-act act-accept"
                  onClick={() => handleSendAction('ACCEPT')}
                  disabled={busy}
                >
                  <CheckCircle2 size={12} />
                  <span>✓ Record Approval</span>
                </button>
                <button
                  type="button"
                  className="btn-human-act act-challenge"
                  onClick={() => handleSendAction('CHALLENGE')}
                  disabled={busy}
                >
                  <AlertTriangle size={12} />
                  <span>⚔ Formal Challenge</span>
                </button>
                <button
                  type="button"
                  className="btn-human-act act-override"
                  onClick={() => handleSendAction('OVERRIDE')}
                  disabled={busy}
                >
                  <Edit3 size={12} />
                  <span>✎ Record override request</span>
                </button>
                <button
                  type="button"
                  className="btn-human-act act-rerun"
                  onClick={() => handleSendAction('RERUN')}
                  disabled={busy}
                >
                  <RotateCcw size={12} />
                  <span>↻ Record rerun request</span>
                </button>
                <button
                  type="button"
                  className="btn-human-act act-escalate"
                  onClick={() => handleSendAction('ESCALATE')}
                  disabled={busy}
                >
                  <ArrowUpCircle size={12} />
                  <span>⇧ Record escalation</span>
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
