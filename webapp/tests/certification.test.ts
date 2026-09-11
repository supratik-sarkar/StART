import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { comparisonRows, datasetIdentities, experimentRuns, experimentTrace, validateBundle } from '../src/features/certification/bundle'
const root = new URL('../public/certification/',import.meta.url)
const index = JSON.parse(readFileSync(new URL('index.json',root),'utf8'))
const bundle = validateBundle(JSON.parse(readFileSync(new URL(index.file,root),'utf8')))
const result=(domain:string)=>bundle.champion_challenger.find(r=>r.domain===domain)!
describe('certification evidence boundaries',()=>{
  it('preserves emitted champion, means, confidence bounds, and backend order',()=>{
    const source=bundle.champion_challenger.find(r=>r.layer==='real_external')!,rows=comparisonRows(source)
    expect(rows[0].mean).toBe(source.champion_mean)
    expect(rows[0].ci95).toEqual(source.champion_ci95)
    expect(rows.map(r=>r.model)).toEqual([source.champion_model,...source.challenger_summaries.map((r:any)=>r.model)])
  })
  it('keeps CNN accuracy and out-of-unit-interval GRU bounds',()=>{
    const rows=comparisonRows(result('deep_learning'))
    expect(rows.find(r=>r.model==='simple_cnn_vision')?.metric).toBe('accuracy')
    expect(rows.find(r=>r.model==='gru_sequence')?.ci95?.[1]).toBe(1.000191)
  })
  it('does not rank portfolio, calibration, or scenario methods',()=>{
    for(const d of ['portfolio','market_risk','scenario_traded_risk'])expect(comparisonRows(result(d))).toEqual([])
  })
  it('never joins golden GPT results to real-data predictive evidence',()=>{
    expect(experimentTrace(bundle,'EXP_PRED_REAL_ADULT')).toBeUndefined()
    expect(experimentRuns(bundle,'EXP_PRED_REAL_ADULT').every(r=>r.layer==='real_external'&&r.policy==='deterministic')).toBe(true)
    expect(experimentTrace(bundle,'EXP_PRED_GOLDEN')?.model).toBe('gpt-4.1')
  })
  it('keeps all five seed records and separates FM from FFM',()=>{
    const runs=experimentRuns(bundle,'EXP_RECOMMENDER')
    for(const m of ['ffm','fm','ncf','mf'])expect(runs.filter(r=>r.model===m&&r.policy==='deterministic').map(r=>r.seed)).toEqual([0,1,2,3,4])
    expect(runs.find(r=>r.model==='ffm')?.preprocessing.field_aware).toBe(true)
    expect(runs.find(r=>r.model==='fm')?.preprocessing.field_aware).toBe(false)
  })
  it('uses sequence and vision identities without borrowing tabular dimensions',()=>{
    const ids=datasetIdentities(bundle,experimentRuns(bundle,'EXP_DEEP_LEARNING'))
    expect(ids.map(d=>d.dataset_id)).toEqual(['deep_learning_v1','sequence_demo_v1','vision_demo_v1'])
    expect(ids.find(d=>d.dataset_id==='vision_demo_v1')?.rows).toBeNull()
  })
  it('retains exclusions and absent sensitivity response rows',()=>{
    expect(bundle.xai_results.filter(r=>r.status==='NOT_EXECUTABLE').map(r=>r.method)).toEqual(['accumulated_local_effects','lime_tabular_explainer'])
    expect(bundle.real_data_domain_coverage.find(r=>r.domain==='recommender')?.real_dataset_executed).toBe(false)
    expect(bundle.sensitivity_results[0].shock_grid).toEqual([-.3,-.2,-.1,-.05,0,.05,.1,.2,.3])
    expect(bundle.sensitivity_results[0].response_rows).toBeUndefined()
  })
  it('fails closed when bundle records are missing',()=>{
    expect(()=>validateBundle({...bundle,deterministic_runs:null})).toThrow('deterministic_runs')
  })
})
