import React, { useEffect, useMemo } from 'react'
import {
  X,
  Pin,
  Trash2,
  ExternalLink,
  Layers,
  FileCheck,
  Shield,
  Columns,
  Clock
} from 'lucide-react'
import type { PinnedItem } from '../../contracts/types'

interface PinnedItemsDrawerProps {
  isOpen: boolean
  onClose: () => void
  pinnedItems: PinnedItem[]
  onUnpin: (id: string) => void
  onSelectArtifact?: (artifactId: string) => void
  onSelectEvidence?: (evidenceId: string) => void
  onCompare?: (runA: string, runB: string) => void
}

export function PinnedItemsDrawer({
  isOpen,
  onClose,
  pinnedItems,
  onUnpin,
  onSelectArtifact,
  onSelectEvidence,
  onCompare,
}: PinnedItemsDrawerProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  // Extract distinct run IDs from pinned items
  const distinctRuns = useMemo(() => {
    const s = new Set<string>()
    pinnedItems.forEach((p) => {
      if (p.runId) s.add(p.runId)
    })
    return Array.from(s)
  }, [pinnedItems])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50 backdrop-blur-xs animate-in fade-in duration-100">
      <div
        className="w-full max-w-md h-full bg-[#121418] border-l border-[#272d36] shadow-2xl flex flex-col font-sans text-[#cfd7e3] animate-in slide-in-from-right duration-150"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pinned-drawer-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#20252e] bg-[#161a22]">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-[#241c1c] text-[#fc8181] border border-[#3d2727]">
              <Pin size={16} />
            </div>
            <div>
              <h2 id="pinned-drawer-title" className="text-sm font-semibold text-[#edf2f7]">
                Pinned Workspace Items ({pinnedItems.length})
              </h2>
              <p className="text-[11px] text-[#8c9bab]">
                Client-persisted artifacts and evidence across review sessions
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-[#718096] hover:text-white rounded-md hover:bg-[#222832] transition-colors"
            aria-label="Close drawer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Action bar for multi-run compare */}
        {distinctRuns.length >= 2 && onCompare && (
          <div className="px-5 py-2.5 bg-[#171c26] border-b border-[#222a38] flex items-center justify-between text-xs">
            <span className="text-[#a0aec0]">
              {distinctRuns.length} distinct runs pinned
            </span>
            <button
              onClick={() => onCompare(distinctRuns[0], distinctRuns[1])}
              className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium bg-[#2b6cb0] hover:bg-[#3182ce] text-white rounded transition-colors"
            >
              <Columns size={12} />
              <span>Compare First 2 Runs</span>
            </button>
          </div>
        )}

        {/* Pinned items list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {pinnedItems.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center px-4">
              <Pin size={32} className="text-[#3c4656] mb-2" />
              <p className="text-xs font-medium text-[#718096]">No items pinned yet</p>
              <p className="text-[11px] text-[#4a5568] mt-1 max-w-xs">
                Click the pin icon on any artifact or evidence record to bookmark it for cross-run review and comparison.
              </p>
            </div>
          )}

          {pinnedItems.map((item) => (
            <div
              key={item.id}
              className="pinned-item-row p-3 rounded-lg border border-[#232934] bg-[#151921] hover:border-[#323b4b] transition-colors space-y-2"
              data-testid={`pinned-item-${item.itemId}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider ${
                      item.itemType === 'artifact'
                        ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40'
                        : item.itemType === 'evidence'
                        ? 'bg-blue-950/60 text-blue-300 border border-blue-800/40'
                        : 'bg-amber-950/60 text-amber-300 border border-amber-800/40'
                    }`}
                  >
                    {item.itemType === 'artifact' && <Layers size={10} />}
                    {item.itemType === 'evidence' && <FileCheck size={10} />}
                    {item.itemType === 'finding' && <Shield size={10} />}
                    {item.itemType}
                  </span>
                  <span className="text-[10px] font-mono text-[#718096]">
                    Run {item.runId.slice(0, 10)}…
                  </span>
                </div>

                <button
                  onClick={() => onUnpin(item.id)}
                  className="pinned-item-remove-btn text-[#718096] hover:text-red-400 transition-colors p-1"
                  title="Unpin item"
                  data-testid={`unpin-btn-${item.itemId}`}
                >
                  <Trash2 size={13} />
                </button>
              </div>

              <div className="text-xs font-medium text-[#edf2f7] leading-snug">
                {item.label}
              </div>

              <div className="flex items-center justify-between text-[11px] text-[#718096] pt-1 border-t border-[#1c222c]">
                <span className="truncate">
                  {item.producerProvenance ? (
                    <span>Producer: {item.producerProvenance}</span>
                  ) : item.stage ? (
                    <span>Stage: {item.stage}</span>
                  ) : (
                    <span>ID: {item.itemId}</span>
                  )}
                </span>

                <div className="flex items-center gap-2">
                  {item.itemType === 'artifact' && onSelectArtifact && (
                    <button
                      onClick={() => onSelectArtifact(item.itemId)}
                      className="text-[#63b3ed] hover:underline flex items-center gap-1 text-[11px]"
                    >
                      <span>View</span>
                      <ExternalLink size={10} />
                    </button>
                  )}
                  {item.itemType === 'evidence' && onSelectEvidence && (
                    <button
                      onClick={() => onSelectEvidence(item.itemId)}
                      className="text-[#63b3ed] hover:underline flex items-center gap-1 text-[11px]"
                    >
                      <span>Inspect</span>
                      <ExternalLink size={10} />
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-[#1c222b] bg-[#101419] text-center text-[11px] text-[#5a687c]">
          Items remain stored locally across sessions
        </div>
      </div>
    </div>
  )
}
