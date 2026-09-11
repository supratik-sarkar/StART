import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  clampSplitRatio,
  resolveInitialSplitRatio,
  DEFAULT_SPLIT_RATIO,
  SPLIT_RATIO_STORAGE_KEY,
} from '../app/splitterLayout'

export type CollapseState = 'none' | 'left' | 'right'

export interface ResizableWorkspaceProps {
  id?: string
  dataTestId?: string
  isRightPaneUnlocked: boolean
  leftContent: React.ReactNode
  rightContent?: React.ReactNode
  ariaLabelLeft?: string
  ariaLabelRight?: string
  className?: string
  splitRatio?: number
  onSplitRatioChange?: (ratio: number) => void
  showPresetsBar?: boolean
}

export function ResizableWorkspace({
  id = 'workspace-split-container',
  dataTestId = 'start-dynamic-workspace',
  isRightPaneUnlocked,
  leftContent,
  rightContent,
  ariaLabelLeft = 'Left Workspace Pane',
  ariaLabelRight = 'Right Workspace Pane',
  className = '',
  splitRatio: controlledRatio,
  onSplitRatioChange,
  showPresetsBar = true,
}: ResizableWorkspaceProps) {
  // Split ratio state
  const [internalRatio, setInternalRatio] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(SPLIT_RATIO_STORAGE_KEY)
      return resolveInitialSplitRatio(saved)
    } catch {
      return DEFAULT_SPLIT_RATIO
    }
  })

  const splitRatio = controlledRatio !== undefined ? controlledRatio : internalRatio
  const [lastNonCollapsedRatio, setLastNonCollapsedRatio] = useState<number>(() => splitRatio)
  const [collapseState, setCollapseState] = useState<CollapseState>('none')
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLElement | null>(null)

  const updateRatio = useCallback(
    (ratio: number) => {
      const clamped = clampSplitRatio(ratio)
      setInternalRatio(clamped)
      setLastNonCollapsedRatio(clamped)
      setCollapseState('none')
      try {
        localStorage.setItem(SPLIT_RATIO_STORAGE_KEY, String(clamped))
      } catch {}
      onSplitRatioChange?.(clamped)
    },
    [onSplitRatioChange]
  )

  const handleMouseDown = (e: React.PointerEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDoubleClick = () => {
    updateRatio(50)
  }

  const collapseLeft = () => {
    if (collapseState === 'left') {
      // Restore
      setCollapseState('none')
      const target = lastNonCollapsedRatio || 50
      setInternalRatio(target)
      try {
        localStorage.setItem(SPLIT_RATIO_STORAGE_KEY, String(target))
      } catch {}
      onSplitRatioChange?.(target)
    } else {
      if (splitRatio > 5 && splitRatio < 95) {
        setLastNonCollapsedRatio(splitRatio)
      }
      setCollapseState('left')
    }
  }

  const collapseRight = () => {
    if (collapseState === 'right') {
      // Restore
      setCollapseState('none')
      const target = lastNonCollapsedRatio || 50
      setInternalRatio(target)
      try {
        localStorage.setItem(SPLIT_RATIO_STORAGE_KEY, String(target))
      } catch {}
      onSplitRatioChange?.(target)
    } else {
      if (splitRatio > 5 && splitRatio < 95) {
        setLastNonCollapsedRatio(splitRatio)
      }
      setCollapseState('right')
    }
  }

  const restore = () => {
    setCollapseState('none')
    const target = lastNonCollapsedRatio || 50
    setInternalRatio(target)
    try {
      localStorage.setItem(SPLIT_RATIO_STORAGE_KEY, String(target))
    } catch {}
    onSplitRatioChange?.(target)
  }

  // Drag listeners
  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const container = containerRef.current || document.getElementById(id)
      if (!container) return
      const rect = container.getBoundingClientRect()
      if (rect.width <= 0) return
      const rawPercent = ((e.clientX - rect.left) / rect.width) * 100
      const clamped = clampSplitRatio(rawPercent)
      setInternalRatio(clamped)
      setLastNonCollapsedRatio(clamped)
      setCollapseState('none')
      onSplitRatioChange?.(clamped)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      setInternalRatio((curr) => {
        try {
          localStorage.setItem(SPLIT_RATIO_STORAGE_KEY, String(curr))
        } catch {}
        return curr
      })
    }

    window.addEventListener('pointermove', handleMouseMove)
    window.addEventListener('pointerup', handleMouseUp)
    return () => {
      window.removeEventListener('pointermove', handleMouseMove)
      window.removeEventListener('pointerup', handleMouseUp)
    }
  }, [isDragging, id, onSplitRatioChange])

  useEffect(() => { if (!isRightPaneUnlocked) setCollapseState('none') }, [isRightPaneUnlocked])

  // Compute pane styles based on split state and collapse state
  const leftPaneStyle: React.CSSProperties = !isRightPaneUnlocked
    ? { flex: '1 1 100%', maxWidth: '100%', width: '100%', borderRight: 'none' }
    : collapseState === 'right'
    ? { flex: '0 0 calc(100% - 10px)', maxWidth: 'calc(100% - 10px)', width: 'calc(100% - 10px)', borderRight: 'none' }
    : collapseState === 'left'
    ? { display: 'none' }
    : {
        flex: `0 0 ${splitRatio}%`,
        maxWidth: `${splitRatio}%`,
        width: `${splitRatio}%`,
      }

  const rightPaneStyle: React.CSSProperties = !isRightPaneUnlocked
    ? { display: 'none' }
    : collapseState === 'left'
    ? { flex: '0 0 calc(100% - 10px)', maxWidth: 'calc(100% - 10px)', width: 'calc(100% - 10px)', borderLeft: 'none' }
    : collapseState === 'right'
    ? { display: 'none' }
    : {
        flex: `0 0 calc(${100 - splitRatio}% - 8px)`,
        maxWidth: `calc(${100 - splitRatio}% - 8px)`,
        width: `calc(${100 - splitRatio}% - 8px)`,
      }

  return (
    <div className="resizable-workspace-wrapper" style={{ display: 'flex', flexDirection: 'column', flex: '1 1 0%', minHeight: 0, height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Presets and Collapse Toolbar (Only when right pane is unlocked) */}
      {isRightPaneUnlocked && showPresetsBar && (
        <div className="split-workspace-bar" data-testid="split-workspace-bar">
          <div className="split-presets-group">
            <span className="presets-label">Layout:</span>
            <button
              type="button"
              className={`preset-btn ${collapseState === 'left' ? 'active' : ''}`}
              onClick={collapseState === 'left' ? restore : collapseLeft}
              title={collapseState === 'left' ? 'Restore previous split ratio' : 'Collapse left pane (100% right)'}
              data-testid="preset-collapse-left"
            >
              {collapseState === 'left' ? 'Restore Split' : '‹ Hide Left'}
            </button>
            <button
              type="button"
              className={`preset-btn ${collapseState === 'none' && splitRatio === 35 ? 'active' : ''}`}
              onClick={() => updateRatio(35)}
              title="Preset 35% Left / 65% Right"
              data-testid="preset-35-65"
            >
              35 / 65
            </button>
            <button
              type="button"
              className={`preset-btn ${collapseState === 'none' && splitRatio === 50 ? 'active' : ''}`}
              onClick={() => updateRatio(50)}
              title="Preset 50% Left / 50% Right (Default)"
              data-testid="preset-50-50"
            >
              50 / 50
            </button>
            <button
              type="button"
              className={`preset-btn ${collapseState === 'none' && splitRatio === 65 ? 'active' : ''}`}
              onClick={() => updateRatio(65)}
              title="Preset 65% Left / 35% Right"
              data-testid="preset-65-35"
            >
              65 / 35
            </button>
            <button
              type="button"
              className={`preset-btn ${collapseState === 'right' ? 'active' : ''}`}
              onClick={collapseState === 'right' ? restore : collapseRight}
              title={collapseState === 'right' ? 'Restore previous split ratio' : 'Collapse right pane (100% left)'}
              data-testid="preset-collapse-right"
            >
              {collapseState === 'right' ? 'Restore Split' : 'Hide Right ›'}
            </button>
          </div>
          <div className="split-hint">
            <span>Double-click divider to reset to 50 / 50 · Drag 5–95%</span>
          </div>
        </div>
      )}

      {/* Main Split Layout */}
      <main
        ref={containerRef as any}
        id={id}
        data-testid={dataTestId}
        className={`workspace-split-container ${isRightPaneUnlocked ? 'has-split' : 'is-single-pane'} ${isDragging ? 'is-dragging' : ''} ${collapseState !== 'none' ? `is-collapsed-${collapseState}` : ''} ${className}`}
      >
        {/* Left Pane */}
        {(!isRightPaneUnlocked || collapseState !== 'left') && (
          <section
            className="workspace-left-pane pane-left-execution porcelain-left-pane"
            aria-label={ariaLabelLeft}
            data-testid="workspace-left-pane"
            style={leftPaneStyle}
          >
            {leftContent}
          </section>
        )}

        {/* Divider / Splitter (Only mounted when right pane is unlocked) */}
        {isRightPaneUnlocked && (
          <div
            className={`workspace-splitter ${isDragging ? 'is-dragging' : ''} ${collapseState !== 'none' ? `is-collapsed-${collapseState}` : ''}`}
            onPointerDown={handleMouseDown}
            onDoubleClick={handleDoubleClick}
            title="Drag to resize panes, double-click to reset to 50/50"
            role="separator"
            aria-label="Resize analytical panes"
            aria-orientation="vertical"
            onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); updateRatio(splitRatio + (e.key === 'ArrowLeft' ? -1 : 1)) } else if (e.key === 'Home') { e.preventDefault(); updateRatio(50) } }}
            aria-valuenow={collapseState === 'none' ? splitRatio : collapseState === 'left' ? 0 : 100}
            aria-valuemin={5}
            aria-valuemax={95}
            tabIndex={0}
            data-testid="workspace-splitter"
          >
            <div className="splitter-thumb" data-testid="splitter-handle" />
          </div>
        )}

        {/* Right Pane */}
        {isRightPaneUnlocked && collapseState !== 'right' && (
          <aside
            className="workspace-right-pane pane-right-output porcelain-right-pane"
            aria-label={ariaLabelRight}
            data-testid="workspace-right-pane"
            style={rightPaneStyle}
          >
            {rightContent}
          </aside>
        )}
      </main>
    </div>
  )
}
