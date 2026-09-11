import { afterEach, describe, expect, it, vi } from 'vitest'
import { liveRequest, mergeDomainRecords, loadLiveCertification } from '../src/features/certification/liveApi'
afterEach(()=>vi.unstubAllGlobals())
describe('live API binding boundaries',()=>{
 it('does not join golden challengers into a real experiment',()=>{
  const rows=mergeDomainRecords([{domain:'predictive',experiment_id:'golden'},{domain:'predictive',experiment_id:'real',champion_mean:.8}],[{domain:'predictive',experiment_id:'golden',challenger_summaries:[{model:'only-golden'}]}])
  expect(rows[0].challenger_summaries).toHaveLength(1)
  expect(rows[1].challenger_summaries).toBeUndefined()
  expect(rows[1].champion_mean).toBe(.8)
  expect(rows[1].presentation_limitation).toContain('BACKEND_BLOCKER')
 })
 it('fails closed instead of falling back to a stale snapshot',async()=>{
  const fetch=vi.fn().mockResolvedValue({ok:false,status:503});vi.stubGlobal('fetch',fetch)
  await expect(loadLiveCertification(new AbortController().signal)).rejects.toThrow('503')
  expect(fetch.mock.calls.every(([url])=>!String(url).includes('/certification/index.json'))).toBe(true)
 })
 it('does not expose validation echoes in an error message',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,status:422,json:async()=>({detail:'sensitive echoed input'})}))
  await expect(liveRequest('/data/provider-sessions',{method:'POST',body:{credentials:{key:'test'}}})).rejects.toThrow('Live API request failed (422)')
 })
 it('preserves zero, false, and NOT_EVALUATED',async()=>{
  const data={worker_count:0,distributed:false,temporal_ordering:'NOT_EVALUATED'}
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({success:true,data})}))
  expect(await liveRequest('/data/runtime')).toEqual(data)
 })
 it('rejects an unsuccessful envelope',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({success:false,data:{status:'CERTIFIED'}})}))
  await expect(liveRequest('/certification')).rejects.toThrow('unsupported response envelope')
 })
})

import { isSelectedRunPresentation } from '../src/features/certification/LiveRunPresentation'
it('rejects other-run and experiment-wide science even when available',()=>{
 const d={requested_run_id:'r',source_run_id:'r',source_scope:'RUN',status:'AVAILABLE'}
 expect(isSelectedRunPresentation(d,'r')).toBe(true)
 expect(isSelectedRunPresentation({...d,source_run_id:'other'},'r')).toBe(false)
 expect(isSelectedRunPresentation({...d,source_scope:'CERTIFICATION_EXPERIMENT'},'r')).toBe(false)
 expect(isSelectedRunPresentation({...d,status:'NOT_AVAILABLE_FOR_RUN'},'r')).toBe(false)
})
it('rejects a mismatched selected experiment in a scoped detail response',()=>{
 const rows=mergeDomainRecords([{domain:'predictive',experiment_id:'real',layer:'real_external'}],[{domain:'predictive',experiment_id:'real',selected_experiment_id:'golden',challenger_summaries:[1]}])
 expect(rows[0].challenger_summaries).toBeUndefined()
})
it('passes an opaque provider session only in the supported header',async()=>{
 const fetch=vi.fn().mockResolvedValue({ok:true,json:async()=>({success:true,data:{}})});vi.stubGlobal('fetch',fetch)
 await liveRequest('/data/resolve',{method:'POST',body:{dataset_id:'public'},providerSessionId:'opaque-test'})
 expect(fetch.mock.calls[0][0]).not.toContain('opaque-test')
 expect(fetch.mock.calls[0][1].headers['X-Provider-Session-ID']).toBe('opaque-test')
})
