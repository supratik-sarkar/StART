import { useEffect, useState } from 'react'
import type { ArtifactRecord, RunSnapshot } from '../../contracts/types'

// Presentation-only projection of the existing open canonical dictionaries.
// No metrics, classifications, configuration defaults or evidence are synthesized here.
export type ScientificRecord = Record<string, any>
export const object = (value: unknown): ScientificRecord => value && typeof value === 'object' && !Array.isArray(value) ? value as ScientificRecord : {}
export const label = (key: string) => ({ roc_auc: 'ROC–AUC', f1: 'F1', rmse: 'RMSE', mae: 'MAE', ndcg_at_10: 'NDCG@10', mrr: 'MRR', ece: 'ECE', C: 'C', tn: 'True negative', fp: 'False positive', fn: 'False negative', tp: 'True positive' }[key] || key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' ').replace(/\b\w/, s => s.toUpperCase()))
export const display = (value: unknown): string => value == null ? 'Not supplied' : typeof value === 'boolean' ? String(value) : String(value)
export const payload = (a?: ArtifactRecord): ScientificRecord => object(a?.content ?? a?.preview?.payload)
export function fromArtifacts(artifacts: ArtifactRecord[]): ScientificRecord {
  const find = (type: string, text: RegExp) => artifacts.find(a => a.artifactType === type || text.test(a.artifactId))
  const profile = payload(find('data_profile_table', /DATA-PROFILE|DATA-QUALITY/))
  const perf = payload(find('metric_summary', /METRIC-SUMMARY|PERF-SUMMARY|QUANT-PERF/))
  const config = payload(find('resolved_configuration', /RESOLVED-CONFIGURATION/))
  return {
    model_family: config.family, technique: config.technique ?? perf.technique, task_type: perf.task_type,
    data_selection: profile.data_selection, data_validation: profile.data_validation, preprocessing: profile.preprocessing,
    split_protocol: profile.split_protocol ?? config.split, resolved_configuration: Object.keys(config).length ? config : undefined,
    metrics: perf.metrics, diagnostics: {
      confusion_matrix: payload(find('confusion_matrix', /CONFUSION/)).confusion_matrix,
      roc_curve: payload(find('roc_curve', /ROC/)), calibration: payload(find('calibration_curve', /CALIBRATION/)),
      ranking: payload(find('ranking', /REC-RANKING/)), cold_start: payload(find('cold_start', /COLDSTART/)),
      beyond_accuracy: payload(find('beyond_accuracy', /BEYOND-ACCURACY/)), weights: payload(find('weights', /WEIGHTS/)),
      risk_contributions: payload(find('risk_contributions', /QUANT-RISK|RISK-CONTRIBUTION/)),
      architecture: payload(find('architecture', /ARCH/)), training_curve: payload(find('training_curve', /TRAINING-CURVE/)),
    },
    structural_analysis: payload(find('feature_importance', /FEATURE-IMPORTANCE|PRED-IMPORTANCE/)),
    sensitivity: payload(find('sensitivity_summary', /SENSITIVITY|ROBUSTNESS/)),
    baseline_comparison: payload(find('baseline_comparison', /BASELINE-COMPARISON/)),
  }
}
export function useCanonicalPresentation(run: RunSnapshot | null, enabled: boolean) {
  const [state, setState] = useState<{ runId: string; result: ScientificRecord | null; error?: string } | null>(null)
  useEffect(() => {
    if (!run || !enabled) return
    const abort = new AbortController()
    fetch(`${import.meta.env.VITE_START_API_BASE || ''}/api/v1/runs/${encodeURIComponent(run.runId)}/presentation`, { signal: abort.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Analytical presentation unavailable (${response.status})`)
        const raw = await response.json()
        const presentation = raw.data?.presentation ?? raw.presentation ?? raw
        const result = presentation?.canonical_analytical_result
        if (result && result.run_id !== run.runId) throw new Error('Analytical presentation run identity does not match')
        setState({ runId: run.runId, result: result ?? null })
      }).catch(error => { if (!abort.signal.aborted) setState({ runId: run.runId, result: null, error: error.message }) })
    return () => abort.abort()
  }, [run?.runId, run?.phase, enabled])
  return state?.runId === run?.runId ? state : null
}

export function isAnalyticalObject(a: ArtifactRecord): boolean {
  const value = a.content ?? a.preview?.payload
  if (value == null) return false
  const type = a.artifactType ?? ''
  if (/configuration|metadata|latex|log/.test(type)) return false
  if (a.mimeType?.startsWith('image/') || a.mimeType === 'application/pdf') return true
  if (['plot','table'].includes(a.kind) || /table|curve|matrix|profile/.test(type)) return typeof value === 'object' && Object.keys(value).length > 0
  return false
}
