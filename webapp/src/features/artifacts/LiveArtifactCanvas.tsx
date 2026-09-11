import { useState, useEffect } from 'react'
import {
  Layers,
  Sparkles,
  ExternalLink,
  Hash,
  Clock,
  CheckCircle2,
  FileText,
  LineChart,
  Pin,
} from 'lucide-react'
import type { ArtifactRecord } from '../../contracts/types'
import { TypedArtifactRenderer } from './TypedArtifactRenderer'
import { formatTimeOnly } from '../../utils/formatTimestamp'

interface LiveArtifactCanvasProps {
  artifacts: ArtifactRecord[]
  selectedArtifactId: string | null
  onSelectArtifact: (id: string) => void
  onHighlightEvidence?: (evidenceId: string) => void
  onHighlightStage?: (stageId: string) => void
  onPinArtifact?: (art: ArtifactRecord) => void
  isArtifactPinned?: (artifactId: string) => boolean
}

export function LiveArtifactCanvas({
  artifacts,
  selectedArtifactId,
  onSelectArtifact,
  onHighlightEvidence,
  onHighlightStage,
  onPinArtifact,
  isArtifactPinned,
}: LiveArtifactCanvasProps) {
  const [lastArrivedId, setLastArrivedId] = useState<string | null>(null)

  // Arrival pulse animation effect
  useEffect(() => {
    if (artifacts.length > 0) {
      const latest = artifacts[artifacts.length - 1]
      setLastArrivedId(latest.artifactId)
      const t = setTimeout(() => setLastArrivedId(null), 1200)
      return () => clearTimeout(t)
    }
  }, [artifacts.length])

  const activeArt = artifacts.find((a) => a.artifactId === selectedArtifactId) || artifacts[0]

  if (artifacts.length === 0) {
    return (
      <div className="live-artifact-canvas empty-canvas" role="region" aria-label="Streaming Artifact Canvas">
        <div className="empty-canvas-content">
          <div className="empty-icon-orb">
            <LineChart size={28} className="text-muted" />
          </div>
          <h3>Streaming Artifact Canvas</h3>
          <p>
            Deterministic plots, calibration diagrams, and confusion surfaces will stream into this canvas in real time
            as execution boundaries complete.
          </p>
        </div>
      </div>
    )
  }

  const producerNode = (activeArt as any)?.producerNodeId || (activeArt as any)?.producing_step_id
  const evidenceIds: string[] = (activeArt as any)?.evidenceIds || (activeArt as any)?.evidence_ids || []
  const dataFingerprint: string = (activeArt as any)?.dataFingerprint || (activeArt as any)?.data_fingerprint || ''

  return (
    <div className="live-artifact-canvas" role="region" aria-label="Streaming Artifact Canvas">
      {/* Canvas Header & Tabs */}
      <div className="artifact-canvas-header">
        <div className="canvas-title-row">
          <div className="canvas-title-group">
            <Layers size={14} className="text-indigo" />
            <span className="canvas-title">Analytical Canvas</span>
          </div>
          <span className="artifact-count-pill">{artifacts.length} Active Surfaces</span>
        </div>

        {/* Artifact Tabs */}
        <div className="artifact-tabs-bar" role="tablist">
          {artifacts.map((art) => {
            const isSelected = activeArt?.artifactId === art.artifactId
            const isNewArrival = lastArrivedId === art.artifactId
            const artKind = art.kind || 'plot'

            return (
              <button
                key={art.artifactId}
                type="button"
                role="tab"
                aria-selected={isSelected}
                className={`artifact-tab-btn ${isSelected ? 'active' : ''} ${isNewArrival ? 'arrival-pulse' : ''}`}
                onClick={() => onSelectArtifact(art.artifactId)}
              >
                <span className="tab-kind-dot" />
                <span className="tab-title">{art.label || art.artifactId}</span>
                {isNewArrival && <Sparkles size={11} className="arrival-icon" />}
              </button>
            )
          })}
        </div>
      </div>

      {/* Active Artifact Container */}
      {activeArt && (
        <div className="artifact-viewport">
          {/* Provenance Banner */}
          <div className="artifact-provenance-banner">
            <div className="provenance-left">
              <div className="provenance-title-row flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h4 className="artifact-heading">{activeArt.label || activeArt.artifactId}</h4>
                  <span className="artifact-type-tag">{activeArt.kind?.toUpperCase() || 'PLOT'}</span>
                </div>
                {onPinArtifact && (
                  <button
                    type="button"
                    className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                      isArtifactPinned?.(activeArt.artifactId)
                        ? 'text-[#fc8181] bg-red-950/40 border border-red-800/40 font-medium'
                        : 'text-[#a0aec0] hover:text-white bg-[#1a202c] border border-[#2d3748]'
                    }`}
                    onClick={() => onPinArtifact(activeArt)}
                    title={isArtifactPinned?.(activeArt.artifactId) ? 'Unpin artifact' : 'Pin artifact to workspace'}
                    data-testid={`pin-artifact-btn-${activeArt.artifactId}`}
                  >
                    <Pin size={11} className={isArtifactPinned?.(activeArt.artifactId) ? 'fill-current' : ''} />
                    <span>{isArtifactPinned?.(activeArt.artifactId) ? 'Pinned' : 'Pin'}</span>
                  </button>
                )}
              </div>
              <p className="artifact-description">{activeArt.description || 'Deterministic review surface.'}</p>
            </div>

            <div className="provenance-meta-grid">
              {producerNode && (
                <div className="provenance-item">
                  <span className="prov-label">Producing Stage:</span>
                  <button
                    type="button"
                    className="prov-link-btn"
                    onClick={() => onHighlightStage?.(producerNode)}
                    title={`Focus producing stage ${producerNode}`}
                  >
                    <Layers size={11} className="mr-1 inline" />
                    <code>{producerNode}</code>
                  </button>
                </div>
              )}

              {dataFingerprint && (
                <div className="provenance-item">
                  <span className="prov-label">Data Fingerprint:</span>
                  <code className="prov-hash" title={dataFingerprint}>
                    <Hash size={10} className="mr-1 inline" />
                    {dataFingerprint.slice(0, 16)}...
                  </code>
                </div>
              )}

              {activeArt.createdAt && (
                <div className="provenance-item">
                  <span className="prov-label">Generated:</span>
                  <span className="prov-time">
                    <Clock size={10} className="mr-1 inline" />
                    {formatTimeOnly(activeArt.createdAt)}
                  </span>
                </div>
              )}
            </div>

            {/* Cited Evidence Records */}
            {evidenceIds.length > 0 && (
              <div className="provenance-evidence-row">
                <span className="prov-label">Cited Evidence:</span>
                <div className="prov-evidence-chips">
                  {evidenceIds.map((eid) => (
                    <button
                      key={eid}
                      type="button"
                      className="prov-ev-chip"
                      onClick={() => onHighlightEvidence?.(eid)}
                      title={`Inspect Evidence ${eid}`}
                    >
                      <span>{eid}</span>
                      <ExternalLink size={9} />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Render Active Artifact Body */}
          <div className="artifact-body-wrapper">
            <TypedArtifactRenderer artifact={activeArt} onHighlightEvidence={onHighlightEvidence} />
          </div>
        </div>
      )}
    </div>
  )
}
