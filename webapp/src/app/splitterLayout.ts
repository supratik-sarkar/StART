/**
 * Dynamic Artifact Canvas & Freeform Splitter Layout Logic
 *
 * Requirements:
 * 1. Old trigger w.artifacts.length > 0 REMOVED.
 * 2. EDA artifacts MUST NOT trigger split.
 * 3. Pre-execution artifacts MUST NOT trigger split.
 * 4. Artifacts from another run MUST NOT trigger split.
 * 5. Dedicated state: runtimeArtifactCanvasUnlocked (false for new live run).
 * 6. New run initial: execution width 100%, right pane ABSENT, splitter ABSENT, reserved space 0.
 * 7. First runtime artifact for THIS active run unlocks canvas (artifact_created event with event.run_id === activeRunId).
 * 8. Historical replay with artifacts unlocks canvas immediately.
 * 9. Continuous freeform splitter across 5%–95% and double-click 50/50 reset preserved.
 */

export const MIN_SPLIT_RATIO = 5
export const MAX_SPLIT_RATIO = 95
export const DEFAULT_SPLIT_RATIO = 50
export const SPLIT_RATIO_STORAGE_KEY = 'start_workspace_split_ratio'

/**
 * Clamp split ratio to continuous [5, 95] range without snapping to presets.
 */
export function clampSplitRatio(ratio: number): number {
  if (isNaN(ratio)) return DEFAULT_SPLIT_RATIO
  const rounded = Math.round(ratio)
  return Math.max(MIN_SPLIT_RATIO, Math.min(MAX_SPLIT_RATIO, rounded))
}

/**
 * Resolve initial split ratio from storage or fallback to 50/50.
 */
export function resolveInitialSplitRatio(savedValue: string | null | undefined): number {
  if (!savedValue) return DEFAULT_SPLIT_RATIO
  const parsed = parseFloat(savedValue)
  if (isNaN(parsed)) return DEFAULT_SPLIT_RATIO
  return clampSplitRatio(parsed)
}

export interface WorkspaceLayout {
  runtimeArtifactCanvasUnlocked: boolean
  isSplitterVisible: boolean
  isArtifactCanvasVisible: boolean
  isPresetsVisible: boolean
  emptyArtifactSpaceReserved: number
  splitRatio: number
  leftPaneStyle: {
    flex: string
    maxWidth: string
    width?: string
    borderRight?: string
  }
  rightPaneStyle: {
    flex: string
    maxWidth: string
  } | null
}

/**
 * Compute the complete workspace layout configuration based strictly on runtimeArtifactCanvasUnlocked.
 */
export function computeWorkspaceLayout(params: {
  runtimeArtifactCanvasUnlocked: boolean
  splitRatio: number
}): WorkspaceLayout {
  const isUnlocked = params.runtimeArtifactCanvasUnlocked
  const ratio = clampSplitRatio(params.splitRatio)

  if (!isUnlocked) {
    return {
      runtimeArtifactCanvasUnlocked: false,
      isSplitterVisible: false,
      isArtifactCanvasVisible: false,
      isPresetsVisible: false,
      emptyArtifactSpaceReserved: 0,
      splitRatio: ratio,
      leftPaneStyle: {
        flex: '1 1 100%',
        maxWidth: '100%',
        width: '100%',
        borderRight: 'none',
      },
      rightPaneStyle: null,
    }
  }

  return {
    runtimeArtifactCanvasUnlocked: true,
    isSplitterVisible: true,
    isArtifactCanvasVisible: true,
    isPresetsVisible: true,
    emptyArtifactSpaceReserved: 0,
    splitRatio: ratio,
    leftPaneStyle: {
      flex: `0 0 ${ratio}%`,
      maxWidth: `${ratio}%`,
    },
    rightPaneStyle: {
      flex: `0 0 calc(${100 - ratio}% - 6px)`,
      maxWidth: `calc(${100 - ratio}% - 6px)`,
    },
  }
}

export function isViewableArtifactForCanvas(artifact: any): boolean {
  if (!artifact) return false

  const kind = String(artifact.kind || '').toLowerCase().trim()
  const mime = String(artifact.mimeType || artifact.mime_type || '').toLowerCase().trim()
  const artType = String(artifact.artifactType || artifact.artifact_type || '').toLowerCase().trim()
  const label = String(artifact.label || artifact.title || artifact.name || artifact.artifactId || '').toLowerCase().trim()

  // 1. Explicit Exclusions: JSON, LaTeX/source, plain text, logs, raw metadata, configs, metrics, decisions
  if (
    kind === 'json' ||
    kind === 'metric' ||
    kind === 'evidence' ||
    kind === 'finding' ||
    kind === 'decision' ||
    kind === 'attestation' ||
    artType === 'json' ||
    artType === 'raw_json' ||
    artType === 'metric' ||
    artType.includes('latex') ||
    artType.includes('metadata') ||
    artType.includes('log') ||
    artType.includes('config') ||
    mime === 'text/plain' ||
    mime.includes('latex') ||
    mime.includes('tex') ||
    mime.includes('x-python') ||
    mime.includes('yaml') ||
    label.endsWith('.json') ||
    label.endsWith('.tex') ||
    label.endsWith('.txt') ||
    label.endsWith('.log')
  ) {
    // If explicitly categorized as visual plot, chart, or table, allow it
    if (
      kind === 'plot' ||
      kind === 'table' ||
      kind === 'chart' ||
      artType === 'plot' ||
      artType === 'table' ||
      artType === 'chart' ||
      artType === 'confusion_matrix'
    ) {
      return true
    }
    return false
  }

  // 2. Explicit Inclusions: image, plot/chart, table, PDF, and other renderable inspection surfaces
  if (
    kind === 'plot' ||
    kind === 'chart' ||
    kind === 'table' ||
    kind === 'pdf' ||
    kind === 'report' ||
    kind === 'image'
  ) {
    return true
  }

  if (
    artType === 'plot' ||
    artType === 'chart' ||
    artType === 'table' ||
    artType === 'summary_table' ||
    artType === 'diagnostic_table' ||
    artType === 'confusion_matrix' ||
    artType === 'image' ||
    artType === 'pdf' ||
    artType === 'report'
  ) {
    return true
  }

  if (
    mime.startsWith('image/') ||
    mime === 'application/pdf' ||
    mime === 'text/csv' ||
    mime === 'text/tab-separated-values'
  ) {
    return true
  }

  // 3. Fallback: visual keywords in label or title
  if (
    label.includes('roc') ||
    label.includes('calibration') ||
    label.includes('importance') ||
    label.includes('confusion') ||
    label.includes('heatmap') ||
    label.includes('diagram') ||
    label.includes('curve') ||
    label.includes('surface') ||
    label.includes('plot') ||
    label.includes('chart')
  ) {
    return true
  }

  return false
}

/**
 * Validates whether an artifact event unlocks the canvas for the active run.
 * EDA, pre-execution, and artifacts from other runs MUST NOT trigger unlock.
 * Only artifacts with a meaningful visual inspection surface unlock the canvas.
 */
export function shouldUnlockOnArtifactEvent(params: {
  eventType: string
  eventRunId: string | undefined
  activeRunId: string | null
  artifact?: any
}): boolean {
  if (params.eventType !== 'artifact_created') return false
  if (!params.eventRunId || !params.activeRunId) return false
  if (params.eventRunId !== params.activeRunId) return false
  if (params.artifact) {
    return isViewableArtifactForCanvas(params.artifact)
  }
  return true
}

/**
 * Checks if loaded historical run contains viewable runtime artifacts belonging to this run.
 */
export function shouldUnlockOnReplay(params: {
  runId: string
  artifacts: Array<{ runId: string; artifactId?: string; [key: string]: any }>
}): boolean {
  return params.artifacts.some(
    (a) =>
      (a.runId === params.runId || (a as any).run_id === params.runId) &&
      isViewableArtifactForCanvas(a)
  )
}

/**
 * Compares artifacts fetched during a live run against the baseline at run start.
 * Only unlocks if there are viewable artifacts belonging to the active run not in baseline.
 */
export function hasNewRuntimeArtifactForActiveRun(params: {
  activeRunId: string | null
  baselineArtifactIds: Set<string>
  artifacts: Array<{ artifactId: string; runId: string; [key: string]: any }>
}): boolean {
  if (!params.activeRunId) return false
  return params.artifacts.some(
    (a) =>
      (a.runId === params.activeRunId || (a as any).run_id === params.activeRunId) &&
      !params.baselineArtifactIds.has(a.artifactId) &&
      isViewableArtifactForCanvas(a)
  )
}

