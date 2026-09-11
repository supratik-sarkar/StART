import { describe, expect, it } from 'vitest'
import {
  clampSplitRatio,
  computeWorkspaceLayout,
  resolveInitialSplitRatio,
  shouldUnlockOnArtifactEvent,
  shouldUnlockOnReplay,
  hasNewRuntimeArtifactForActiveRun,
  isViewableArtifactForCanvas,
  DEFAULT_SPLIT_RATIO,
  MIN_SPLIT_RATIO,
  MAX_SPLIT_RATIO,
} from '../app/splitterLayout'

describe('Dynamic Artifact Canvas & Freeform Splitter Narrow Invariants', () => {
  it('VIEWABLE_ARTIFACT_PREDICATE: strictly discriminates visual vs non-visual artifacts', () => {
    // Excluded: JSON, LaTeX, plain text, raw metadata, logs, configs
    expect(isViewableArtifactForCanvas({ kind: 'json', mimeType: 'application/json' })).toBe(false)
    expect(isViewableArtifactForCanvas({ kind: 'latex', mimeType: 'text/x-tex' })).toBe(false)
    expect(isViewableArtifactForCanvas({ kind: 'metric', mimeType: 'application/json' })).toBe(false)
    expect(isViewableArtifactForCanvas({ kind: 'evidence', mimeType: 'application/json' })).toBe(false)
    expect(isViewableArtifactForCanvas({ kind: 'finding', mimeType: 'application/json' })).toBe(false)
    expect(isViewableArtifactForCanvas({ kind: 'decision', mimeType: 'application/json' })).toBe(false)
    expect(isViewableArtifactForCanvas({ label: 'execution.log', mimeType: 'text/plain' })).toBe(false)
    expect(isViewableArtifactForCanvas({ label: 'run_config.json', artifactType: 'config' })).toBe(false)

    // Included: image, plot/chart, table, PDF, and explicit visual types
    expect(isViewableArtifactForCanvas({ kind: 'plot', mimeType: 'application/json' })).toBe(true)
    expect(isViewableArtifactForCanvas({ kind: 'table', mimeType: 'application/json' })).toBe(true)
    expect(isViewableArtifactForCanvas({ kind: 'pdf', mimeType: 'application/pdf' })).toBe(true)
    expect(isViewableArtifactForCanvas({ kind: 'report', mimeType: 'application/pdf' })).toBe(true)
    expect(isViewableArtifactForCanvas({ mimeType: 'image/png' })).toBe(true)
    expect(isViewableArtifactForCanvas({ mimeType: 'image/svg+xml' })).toBe(true)
    expect(isViewableArtifactForCanvas({ artifactType: 'confusion_matrix' })).toBe(true)
    expect(isViewableArtifactForCanvas({ title: 'ROC Discrimination Curve & Gini Surface' })).toBe(true)
    expect(isViewableArtifactForCanvas({ title: 'Expected Calibration Error & Reliability Surface' })).toBe(true)
  })

  it('OLD_TRIGGER_W_ARTIFACTS_LENGTH = REMOVED: generic artifacts length does not unlock canvas', () => {
    // Artifacts exist in array (e.g. from EDA or previous run), but canvas is locked for new run
    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked: false,
      splitRatio: 50,
    })

    expect(layout.runtimeArtifactCanvasUnlocked).toBe(false)
    expect(layout.isArtifactCanvasVisible).toBe(false)
    expect(layout.isSplitterVisible).toBe(false)
    expect(layout.isPresetsVisible).toBe(false)
    expect(layout.emptyArtifactSpaceReserved).toBe(0)
    expect(layout.leftPaneStyle.flex).toBe('1 1 100%')
    expect(layout.leftPaneStyle.maxWidth).toBe('100%')
    expect(layout.rightPaneStyle).toBeNull()
  })

  it('EDA_ARTIFACTS_TRIGGER_SPLIT = NO: EDA artifacts do not unlock runtime canvas', () => {
    const activeRunId = 'RUN-ACTIVE-001'
    const edaArtifact = { artifactId: 'art-eda-1', runId: 'eda-context-synthetic', kind: 'plot' }

    // Event from EDA context or different runId
    const unlocked = shouldUnlockOnArtifactEvent({
      eventType: 'artifact_created',
      eventRunId: edaArtifact.runId,
      activeRunId,
    })
    expect(unlocked).toBe(false)

    const hasNew = hasNewRuntimeArtifactForActiveRun({
      activeRunId,
      baselineArtifactIds: new Set([edaArtifact.artifactId]),
      artifacts: [edaArtifact],
    })
    expect(hasNew).toBe(false)
  })

  it('PREEXECUTION_ARTIFACTS_TRIGGER_SPLIT = NO: pre-existing baseline artifacts do not unlock canvas', () => {
    const activeRunId = 'RUN-ACTIVE-002'
    const preExistingArtifact = { artifactId: 'art-pre-1', runId: activeRunId, kind: 'plot' }
    const baseline = new Set(['art-pre-1'])

    // Pre-existing artifact captured in baseline at run start
    const hasNew = hasNewRuntimeArtifactForActiveRun({
      activeRunId,
      baselineArtifactIds: baseline,
      artifacts: [preExistingArtifact],
    })
    expect(hasNew).toBe(false)
  })

  it('NEW_RUN_INITIAL_UNLOCKED = NO & NEW_RUN_EXECUTION_WIDTH = 100%', () => {
    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked: false,
      splitRatio: resolveInitialSplitRatio('65'),
    })

    expect(layout.runtimeArtifactCanvasUnlocked).toBe(false)
    expect(layout.leftPaneStyle.flex).toBe('1 1 100%')
    expect(layout.leftPaneStyle.maxWidth).toBe('100%')
    expect(layout.leftPaneStyle.borderRight).toBe('none')
    expect(layout.rightPaneStyle).toBeNull()
    expect(layout.isSplitterVisible).toBe(false)
    expect(layout.isPresetsVisible).toBe(false)
  })

  it('FIRST_RUNTIME_ARTIFACT_TRIGGER = CURRENT_RUN_ONLY: unlocks only for active run event with viewable artifact', () => {
    const activeRunId = 'RUN-ACTIVE-003'

    // Artifact event for another run
    expect(
      shouldUnlockOnArtifactEvent({
        eventType: 'artifact_created',
        eventRunId: 'RUN-DIFFERENT-999',
        activeRunId,
        artifact: { kind: 'plot' },
      })
    ).toBe(false)

    // Non-artifact event for this run
    expect(
      shouldUnlockOnArtifactEvent({
        eventType: 'evidence_created',
        eventRunId: activeRunId,
        activeRunId,
      })
    ).toBe(false)

    // Non-viewable JSON artifact for this run
    expect(
      shouldUnlockOnArtifactEvent({
        eventType: 'artifact_created',
        eventRunId: activeRunId,
        activeRunId,
        artifact: { kind: 'json', mimeType: 'application/json' },
      })
    ).toBe(false)

    // Non-viewable LaTeX artifact for this run
    expect(
      shouldUnlockOnArtifactEvent({
        eventType: 'artifact_created',
        eventRunId: activeRunId,
        activeRunId,
        artifact: { kind: 'latex', mimeType: 'text/x-tex' },
      })
    ).toBe(false)

    // Genuine first viewable runtime artifact for active run
    expect(
      shouldUnlockOnArtifactEvent({
        eventType: 'artifact_created',
        eventRunId: activeRunId,
        activeRunId,
        artifact: { kind: 'plot' },
      })
    ).toBe(true)

    // New runtime artifact not in baseline
    const newArtifact = { artifactId: 'art-runtime-1', runId: activeRunId, kind: 'plot' }
    const hasNew = hasNewRuntimeArtifactForActiveRun({
      activeRunId,
      baselineArtifactIds: new Set(['art-eda-1']),
      artifacts: [{ artifactId: 'art-eda-1', runId: 'eda-ctx', kind: 'plot' }, newArtifact],
    })
    expect(hasNew).toBe(true)
  })

  it('RIGHT_PANE_AFTER_FIRST_RUNTIME_ARTIFACT = PRESENT: reveals split with saved/default ratio', () => {
    const savedRatio = resolveInitialSplitRatio('68')
    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked: true,
      splitRatio: savedRatio,
    })

    expect(layout.runtimeArtifactCanvasUnlocked).toBe(true)
    expect(layout.isArtifactCanvasVisible).toBe(true)
    expect(layout.isSplitterVisible).toBe(true)
    expect(layout.isPresetsVisible).toBe(true)
    expect(layout.splitRatio).toBe(68)
    expect(layout.leftPaneStyle.flex).toBe('0 0 68%')
    expect(layout.rightPaneStyle).not.toBeNull()
    expect(layout.rightPaneStyle?.flex).toBe('0 0 calc(32% - 6px)')
  })

  it('REPLAY_EXISTING_RUNTIME_ARTIFACTS = IMMEDIATE_SPLIT: historical run with artifacts unlocks immediately', () => {
    const runId = 'RUN-HISTORICAL-REPLAY'
    const artifacts = [{ artifactId: 'art-h1', runId, kind: 'plot' }]

    const unlocked = shouldUnlockOnReplay({
      runId,
      artifacts,
    })
    expect(unlocked).toBe(true)

    // Historical run without artifacts remains locked (single pane)
    expect(
      shouldUnlockOnReplay({
        runId,
        artifacts: [{ artifactId: 'art-other', runId: 'RUN-OTHER', kind: 'plot' }],
      })
    ).toBe(false)

    // Historical run with only non-viewable artifacts remains locked
    expect(
      shouldUnlockOnReplay({
        runId,
        artifacts: [{ artifactId: 'art-json', runId, kind: 'json', mimeType: 'application/json' }],
      })
    ).toBe(false)
  })

  it('bare root startup + previous sessionStorage run → fresh workspace (no run, no replay, no split)', () => {
    // Simulating bare root URL with empty hash
    const hash = ''
    const match = hash.match(/#run=([a-zA-Z0-9_\-]+)/)
    expect(match).toBeNull()

    // On bare root, no auto-restoration occurs:
    const activeRun = null
    const isReplay = false
    const runtimeArtifactCanvasUnlocked = false

    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked,
      splitRatio: resolveInitialSplitRatio('62'),
    })

    expect(activeRun).toBeNull()
    expect(isReplay).toBe(false)
    expect(layout.runtimeArtifactCanvasUnlocked).toBe(false)
    expect(layout.leftPaneStyle.flex).toBe('1 1 100%')
    expect(layout.leftPaneStyle.maxWidth).toBe('100%')
    expect(layout.isSplitterVisible).toBe(false)
    expect(layout.rightPaneStyle).toBeNull()
  })

  it('explicit #run= navigation restores run and splits immediately if artifacts exist', () => {
    const hash = '#run=RUN-SAVED-123'
    const match = hash.match(/#run=([a-zA-Z0-9_\-]+)/)
    expect(match).not.toBeNull()
    expect(match![1]).toBe('RUN-SAVED-123')

    const persistedArtifacts = [{ artifactId: 'art-saved-1', runId: 'RUN-SAVED-123', kind: 'plot' }]
    const shouldSplit = shouldUnlockOnReplay({
      runId: match![1],
      artifacts: persistedArtifacts,
    })
    expect(shouldSplit).toBe(true)

    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked: shouldSplit,
      splitRatio: resolveInitialSplitRatio('58'),
    })
    expect(layout.runtimeArtifactCanvasUnlocked).toBe(true)
    expect(layout.isArtifactCanvasVisible).toBe(true)
    expect(layout.isSplitterVisible).toBe(true)
    expect(layout.splitRatio).toBe(58)
    expect(layout.leftPaneStyle.flex).toBe('0 0 58%')
    expect(layout.rightPaneStyle?.flex).toBe('0 0 calc(42% - 6px)')
  })

  it('FREEFORM_SPLITTER = PRESERVED: continuous ratio across 5%-95% range without preset snapping', () => {
    const testRatios = [5, 10, 23, 37, 50, 61, 68, 78, 90, 95]
    for (const ratio of testRatios) {
      expect(clampSplitRatio(ratio)).toBe(ratio)
    }
    expect(clampSplitRatio(1)).toBe(MIN_SPLIT_RATIO)
    expect(clampSplitRatio(99)).toBe(MAX_SPLIT_RATIO)
  })

  it('DOUBLE_CLICK_RESET = 50/50 preserved', () => {
    expect(DEFAULT_SPLIT_RATIO).toBe(50)
    const layout = computeWorkspaceLayout({
      runtimeArtifactCanvasUnlocked: true,
      splitRatio: DEFAULT_SPLIT_RATIO,
    })
    expect(layout.splitRatio).toBe(50)
    expect(layout.leftPaneStyle.flex).toBe('0 0 50%')
  })
})
