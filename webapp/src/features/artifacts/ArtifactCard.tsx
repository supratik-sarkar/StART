import React, { useState } from 'react'
import { Check, Copy, Maximize2, Minimize2, Sparkles, Database, Info } from 'lucide-react'
import type { EdaProvenance } from '../../contracts/types'

export interface ArtifactCardProps {
  id: string
  title: string
  subtitle?: string
  badge?: string
  badgeVariant?: 'default' | 'sage' | 'amber' | 'accent' | 'muted'
  provenance?: EdaProvenance
  children: React.ReactNode
  rawJson?: any
  actions?: React.ReactNode
  defaultExpanded?: boolean
}

export function ArtifactCard({
  id,
  title,
  subtitle,
  badge,
  badgeVariant = 'default',
  provenance,
  children,
  rawJson,
  actions,
  defaultExpanded = false,
}: ArtifactCardProps) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!rawJson && !provenance) return
    const content = JSON.stringify(rawJson || provenance, null, 2)
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const badgeClass = `artifact-badge badge-${badgeVariant}`

  return (
    <>
      <article className="artifact-porcelain-card" id={`artifact-${id}`}>
        {/* Card Header */}
        <header className="artifact-card-header">
          <div className="artifact-title-group">
            <span className="artifact-card-icon">
              <Database size={14} />
            </span>
            <div>
              <h3 className="artifact-card-title">{title}</h3>
              {subtitle && <p className="artifact-card-subtitle">{subtitle}</p>}
            </div>
          </div>
          <div className="artifact-header-actions">
            {badge && <span className={badgeClass}>{badge}</span>}
            <button
              className="icon-action-btn"
              onClick={() => setIsModalOpen(true)}
              title="Expand artifact"
              aria-label="Expand artifact"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </header>

        {/* Card Body */}
        <div className="artifact-card-body">
          {children}
        </div>

        {/* Card Footer with Provenance */}
        <footer className="artifact-card-footer">
          <div className="provenance-tag">
            <Info size={11} />
            <span>
              {provenance
                ? `${provenance.profiling_operation} · ${provenance.sha256_fingerprint || 'seed ' + provenance.seed}`
                : 'Deterministic dataset profiling'}
            </span>
          </div>
          <div className="artifact-footer-actions">
            {actions}
            {rawJson && (
              <button
                className="quiet-copy-btn"
                onClick={handleCopy}
                title="Copy artifact data"
              >
                {copied ? <Check size={11} className="text-sage" /> : <Copy size={11} />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            )}
          </div>
        </footer>
      </article>

      {/* Expanded Modal */}
      {isModalOpen && (
        <div className="artifact-modal-backdrop" onClick={() => setIsModalOpen(false)}>
          <div className="artifact-modal-window" onClick={e => e.stopPropagation()}>
            <header className="artifact-modal-header">
              <div>
                <h3>{title}</h3>
                {subtitle && <p>{subtitle}</p>}
              </div>
              <div className="modal-header-controls">
                {badge && <span className={badgeClass}>{badge}</span>}
                <button className="icon-action-btn" onClick={() => setIsModalOpen(false)}>
                  <Minimize2 size={14} />
                </button>
              </div>
            </header>
            <div className="artifact-modal-body">
              {children}
              {provenance && (
                <div className="modal-provenance-box">
                  <h4>Inspectable Provenance Metadata</h4>
                  <pre className="provenance-pre">
                    {JSON.stringify(provenance, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
