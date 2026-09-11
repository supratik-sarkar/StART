import React, { useState, useEffect, useCallback } from 'react'
import {
  ShieldCheck,
  Bot,
  Sparkles,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Loader2,
  HelpCircle,
  ChevronRight,
  DownloadCloud,
} from 'lucide-react'

export type ReviewMode = 'deterministic' | 'interactive'

export type WebLLMCapability = 'NOT_PROBED' | 'PROBING' | 'SUPPORTED' | 'UNSUPPORTED'

export type InteractiveRuntimeState =
  | 'NOT_PROBED'
  | 'PROBING'
  | 'AVAILABLE'
  | 'MODEL_NOT_LOADED'
  | 'DOWNLOADING'
  | 'LOADING'
  | 'READY'
  | 'UNSUPPORTED'
  | 'ERROR'
  | 'BLOCKED'

export function resolveInteractiveState(
  capability: WebLLMCapability,
  reviewerStatus: string = 'idle',
  isBlocked: boolean = false
): InteractiveRuntimeState {
  if (isBlocked) return 'BLOCKED'
  if (capability === 'NOT_PROBED') return 'NOT_PROBED'
  if (capability === 'PROBING') return 'PROBING'
  if (capability === 'UNSUPPORTED') return 'UNSUPPORTED'

  // capability === 'SUPPORTED'
  const status = reviewerStatus.toLowerCase()
  if (status.includes('failed') || status.includes('error')) return 'ERROR'
  if (status.includes('download') || status.includes('fetching')) return 'DOWNLOADING'
  if (status.includes('loading') || status.includes('initializing') || status.includes('%')) return 'LOADING'
  if (status.includes('ready') || status.includes('smollm2')) return 'READY'
  if (status === 'idle' || !status) return 'MODEL_NOT_LOADED'
  return 'AVAILABLE'
}

interface InteractiveModeSelectorProps {
  currentMode: ReviewMode
  onModeChange: (mode: ReviewMode) => void
  reviewerStatus?: string
  reviewerBusy?: boolean
  onInitReviewer?: () => void
  onRunReview?: () => void
  isBlocked?: boolean
}

export const InteractiveModeSelector: React.FC<InteractiveModeSelectorProps> = ({
  currentMode,
  onModeChange,
  reviewerStatus = 'idle',
  reviewerBusy = false,
  onInitReviewer,
  onRunReview,
  isBlocked = false,
}) => {
  const [capability, setCapability] = useState<WebLLMCapability>('NOT_PROBED')
  const [probeDetails, setProbeDetails] = useState<string>('')

  // Asynchronously probe WebGPU capability without blocking render or gating visibility
  useEffect(() => {
    let cancelled = false
    const probe = async () => {
      setCapability('PROBING')
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        if (!cancelled) {
          setCapability('UNSUPPORTED')
          setProbeDetails('Browser does not support WebGPU API')
        }
        return
      }
      try {
        const adapter = await (navigator as any).gpu.requestAdapter()
        if (!cancelled) {
          if (adapter) {
            setCapability('SUPPORTED')
            setProbeDetails('Hardware WebGPU Acceleration Available')
          } else {
            setCapability('UNSUPPORTED')
            setProbeDetails('No suitable WebGPU adapter found')
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          setCapability('UNSUPPORTED')
          setProbeDetails(err?.message || 'WebGPU initialization failed')
        }
      }
    }
    probe()
    return () => {
      cancelled = true
    }
  }, [])

  const handleSelectMode = useCallback(
    (mode: ReviewMode) => {
      onModeChange(mode)
      if (mode === 'interactive' && reviewerStatus === 'idle' && onInitReviewer && capability === 'SUPPORTED') {
        onInitReviewer()
      }
    },
    [onModeChange, reviewerStatus, onInitReviewer, capability]
  )

  const runtimeState = resolveInteractiveState(capability, reviewerStatus, isBlocked)

  return (
    <div className="interactive-mode-selector" data-testid="interactive-mode-selector">
      {/* Mode Toggle Bar - Always visible regardless of hardware or model status */}
      <div className="mode-toggle-group" role="radiogroup" aria-label="Review Engine Mode">
        <button
          type="button"
          role="radio"
          aria-checked={currentMode === 'deterministic'}
          className={`mode-toggle-btn ${currentMode === 'deterministic' ? 'active' : ''}`}
          onClick={() => handleSelectMode('deterministic')}
          title="Deterministic Review: Formal mathematical evaluation, zero stochastic variance, bitwise reproducible"
          data-testid="mode-btn-deterministic"
        >
          <ShieldCheck size={14} className="text-indigo-400" />
          <span className="mode-label">Deterministic Review</span>
          <span className="mode-badge-audit">Audit-Grade</span>
        </button>

        <button
          type="button"
          role="radio"
          aria-checked={currentMode === 'interactive'}
          className={`mode-toggle-btn ${currentMode === 'interactive' ? 'active' : ''}`}
          onClick={() => handleSelectMode('interactive')}
          title="Interactive Review: Grounded client-side WebLLM dialogue over active EvidenceRecords"
          data-testid="mode-btn-interactive"
        >
          <Bot size={14} className="text-emerald-400" />
          <span className="mode-label">Interactive Review</span>
          <span className="mode-badge-ai">AI Layer</span>
        </button>
      </div>

      {/* Decoupled WebLLM Runtime Status Pill */}
      <div className="webllm-status-strip" data-testid="webllm-status-strip">
        {runtimeState === 'NOT_PROBED' && (
          <div className="status-pill status-probing" title="Hardware probe initializing">
            <span>Checking AI Hardware...</span>
          </div>
        )}

        {runtimeState === 'PROBING' && (
          <div className="status-pill status-probing" title="Probing local WebGPU support">
            <Loader2 size={11} className="animate-spin text-slate-400" />
            <span>Probing WebGPU...</span>
          </div>
        )}

        {runtimeState === 'UNSUPPORTED' && (
          <div
            className="status-pill status-unsupported"
            title={`WebGPU Unavailable: ${probeDetails}. Deterministic review remains fully operational.`}
          >
            <AlertCircle size={11} className="text-amber-400" />
            <span>WebGPU Fallback (CPU)</span>
          </div>
        )}

        {runtimeState === 'MODEL_NOT_LOADED' && (
          <div className="status-pill status-idle" title="WebGPU ready; browser LLM not loaded yet">
            <Cpu size={11} className="text-indigo-400" />
            <span>WebGPU Ready</span>
            {onInitReviewer && (
              <button
                type="button"
                className="init-llm-btn"
                disabled={reviewerBusy}
                onClick={(e) => {
                  e.stopPropagation()
                  onInitReviewer()
                }}
                title="Initialize local SmolLM2 weights in browser"
                data-testid="init-local-webllm-btn"
              >
                <span>Init AI</span>
              </button>
            )}
          </div>
        )}

        {runtimeState === 'DOWNLOADING' && (
          <div className="status-pill status-loading" title={reviewerStatus}>
            <DownloadCloud size={11} className="animate-bounce text-indigo-400" />
            <span>Downloading Model...</span>
          </div>
        )}

        {runtimeState === 'LOADING' && (
          <div className="status-pill status-loading" title={reviewerStatus}>
            <Loader2 size={11} className="animate-spin text-indigo-400" />
            <span>Loading Model...</span>
          </div>
        )}

        {runtimeState === 'READY' && (
          <div className="status-pill status-ready" title="SmolLM2 WebLLM ready for contextual queries">
            <CheckCircle2 size={11} className="text-emerald-400" />
            <span>SmolLM2 Local AI Active</span>
          </div>
        )}

        {runtimeState === 'ERROR' && (
          <div className="status-pill status-error" title={reviewerStatus}>
            <AlertCircle size={11} className="text-rose-400" />
            <span>Model Load Warning</span>
          </div>
        )}

        {runtimeState === 'BLOCKED' && (
          <div className="status-pill status-unsupported" title="Interactive review runtime blocked">
            <AlertCircle size={11} className="text-amber-400" />
            <span>Runtime Blocked</span>
          </div>
        )}

        {runtimeState === 'AVAILABLE' && (
          <div className="status-pill status-ready" title="WebGPU Available">
            <CheckCircle2 size={11} className="text-indigo-400" />
            <span>WebGPU Available</span>
          </div>
        )}
      </div>
    </div>
  )
}
