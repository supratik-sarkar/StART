import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { runConfiguration, compatibleContexts, capabilityLabel } from './runtimeSupport'
import { CapabilityDetails } from './CapabilityDetails'
import { PublicStARTBackend } from '../../adapters/public/PublicStARTBackend'
import { SensitivityResults, sensitivitySeries, TuningResults } from '../investigation/ExperimentResults'
import { WorkbenchHeader } from '../../components/WorkbenchHeader'
import type { ExecutionContext, ScenarioItem, StARTCapabilityManifest } from '../../contracts/types'

describe('Public runtime capability boundary',()=>{
 it('sends seed zero at the top level and only consumed tuning parameters',()=>{expect(runConfiguration('hyperparameter_tuning',0,7)).toEqual({executionMode:'hybrid_workbench',seed:0,parameters:{trials:7}});expect(runConfiguration('predictive_ml',42,7)).toEqual({executionMode:'hybrid_workbench',seed:42,parameters:{}})})
 it('leaves absent values to backend defaults',()=>expect(runConfiguration('predictive_ml')).toEqual({executionMode:'hybrid_workbench',parameters:{}}))
 it('rejects values the backend would clamp or cannot seed',()=>{for(const t of [4,31,7.5,NaN])expect(()=>runConfiguration('hyperparameter_tuning',42,t)).toThrow();expect(()=>runConfiguration('predictive_ml',-1)).toThrow()})
 it('does not substitute an inspection scenario for an execution context',()=>{const cs=[{id:'real'},{id:'other'}] as ExecutionContext[];const ss=[{id:'preview',compatible_context_id:'other',compatible_workflows:['predictive_ml']},{id:'real',compatible_context_id:'real',compatible_workflows:['predictive_ml']}] as ScenarioItem[];expect(compatibleContexts(cs,ss,'predictive_ml').map(c=>c.id)).toEqual(['real'])})
 it('distinguishes FM and FFM',()=>expect(capabilityLabel('factorization_machine')).not.toEqual(capabilityLabel('field_aware_factorization_machine')))
 it('never invents a provider ready state',()=>{const html=renderToStaticMarkup(<WorkbenchHeader mode="DEFINE" onRuntime={()=>{}}/>);expect(html).toContain('Unverified');expect(html).not.toContain('Ready')})
 it('renders only supplied declarations without pretending they are executable controls',()=>{const manifest={domains:{predictive_ml:{name:'Predictive',primary_classification_models:['lightgbm']}}} as unknown as StARTCapabilityManifest;const html=renderToStaticMarkup(<CapabilityDetails manifest={manifest}/>);expect(html).toContain('LightGBM');expect(html).not.toContain('XGBoost');expect(html).not.toContain('<select');expect(html).toContain('read-only declarations')})
 it('does not manufacture the nine sensitivity observations',()=>{expect(sensitivitySeries({rows:[{feature:'age',shock:0,metric:.78},{feature:'age',shock:.3,metric:.7}]})).toEqual([{name:'age',x:[0,.3],y:[.78,.7]}]);expect(renderToStaticMarkup(<SensitivityResults data={{delta_auc:.01}}/>)).toContain('Shock response curves not supplied')})
})

describe('Persisted tuning observations',()=>{
 afterEach(()=>vi.unstubAllGlobals())
 it('loads actual persisted events and rejects records from another run',async()=>{
  const event={event_id:'e1',run_id:'run-1',event_type:'tuning_trial',status:'OK',metadata:{trial:1,validation_metric:.72}}
  const fetcher=vi.fn().mockResolvedValue(new Response(JSON.stringify({success:true,data:{events:[event,{...event,run_id:'other'}]}})))
  vi.stubGlobal('fetch',fetcher)
  const events=await new PublicStARTBackend().getRunEvents('run-1')
  expect(events).toHaveLength(1)
  expect(events[0].metadata?.validation_metric).toBe(.72)
  const html=renderToStaticMarkup(<TuningResults data={undefined} events={events}/>)
  expect(html).toContain('0.72');expect(html).toContain('e1');expect(html).not.toContain('ROC-AUC')
 })
})

describe('Final scientific request binding',()=>{
 it('preserves nested scientific selections and agentic mode exactly',()=>{
 const parameters={model:'lightgbm',hyperparameters:{num_leaves:15},preprocessing:{imputation:'mean',scaler:'minmax'},split:{strategy:'random',test_size:.3},sensitivity:{mode:'parallel_basket'},sensitivity_mode:'parallel_basket'}
 expect(runConfiguration('predictive_ml',0,undefined,parameters,'agentic_session')).toEqual({seed:0,executionMode:'agentic_session',parameters})
 })
 it('never maps FFM to FM',()=>expect(runConfiguration('recommender_system',undefined,undefined,{algorithm:'field_aware_factorization_machine'}).parameters.algorithm).toBe('field_aware_factorization_machine'))
 it('submits plan and run using real aliases and preserves proposal metadata',async()=>{
 const {validateAgentPlanPreview}=await import('../../contracts/validators')
 const p=validateAgentPlanPreview({workflowId:'predictive_ml',contextId:'x',goal:'test',plan:[],executionMode:'agentic_session',agentProposal:{recommended_estimator:'lightgbm'}})
 expect(p.executionMode).toBe('agentic_session');expect(p.agentProposal?.recommended_estimator).toBe('lightgbm')
 })
})

describe('Shock plot labeling',()=>{
 it('retains fractional shock tick precision',()=>{
 const html=renderToStaticMarkup(<SensitivityResults data={{metric_name:'auc_roc',rows:[{feature:'basket',shock:-.3,metric:.7},{feature:'basket',shock:.3,metric:.8}]}}/>)
 expect(html).toContain('-0.30');expect(html).toContain('0.30')
 })
})
