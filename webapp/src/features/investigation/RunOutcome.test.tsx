import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { validateEvidenceRecords } from '../../contracts/validators'
import type { GovernanceState, RunSnapshot, RuntimeEvent } from '../../contracts/types'
import { evidenceStatusSummary, RunOutcome, runDecisionReceipts } from './RunOutcome'

const run: RunSnapshot = {runId:'RUN-1',workflowId:'predictive_ml',contextId:'dataset-1',goal:'Evaluate an existing model',phase:'completed',statusLabel:'Completed',startedAt:'',updatedAt:'',elapsedMs:0,plan:[]}
const raw = (id:string,status:string) => ({evidence_id:id,test_id:'treasury.'+id,test_name:'Pre-registered study',run_id:'RUN-1',status,metrics:{nominal_coverage:0.635},thresholds:[{metric:'nominal_coverage',direction:'lower',fail:0.9}],timestamp:'2026-09-10T08:00:00Z',input_artifact_hash:'sha256-source'})

describe('outcome and evidence truth',()=>{
 it('keeps canonical FAIL distinct from WARN and preserves exact backend metrics and criteria',()=>{
  const records=validateEvidenceRecords([raw('CEV','FAIL'),raw('Stanton','WARN')])
  expect(records[0].status).toBe('FAIL')
  expect(records[0].metrics[0].value).toBe(0.635)
  expect((records[0].metrics[0] as any).status).toBeUndefined()
  expect(records[0].rawSource?.thresholds).toEqual([{metric:'nominal_coverage',direction:'lower',fail:0.9}])
 expect(evidenceStatusSummary(records).map(group=>group.status)).toEqual(['FAIL','WARN'])
 })
 it('retains lowercase backend failure semantics in the evidence summary',()=>{
  const records=validateEvidenceRecords([raw('drift','fail'),raw('warning','warn'),raw('record','recorded')])
  expect(evidenceStatusSummary(records).map(group=>group.status)).toEqual(['FAIL','WARN','RECORDED'])
  expect(records[0].rawSource?.status).toBe('fail')
 })
 it('renders failed evidence and conditional governance without inventing missing agent or grounding states',()=>{
  const evidence=validateEvidenceRecords([raw('CEV','FAIL')])
  const governance:GovernanceState={disposition:'ACCEPT_WITH_CONDITIONS',policyDecision:'WARN',unresolvedItems:['Independent remediation required']}
  const markup=renderToStaticMarkup(<RunOutcome run={run} evidence={evidence} governance={governance} attestation={null} decisions={[]} events={[]} onNavigate={()=>{}}/>)
  expect(markup).toContain('FAIL · treasury.CEV · CEV')
  expect(markup).toContain('Independent remediation required')
  expect(markup).toContain('No structured decision trace was supplied')
 expect(markup).toContain('No run-level claim grounding result was supplied')
 })
 it('keeps a recorded challenge visible after governance moves to pending review',()=>{
  const governance:GovernanceState={disposition:'CHALLENGED_PENDING_REVIEW',unresolvedItems:[]}
  const markup=renderToStaticMarkup(<RunOutcome run={run} evidence={[]} governance={governance} attestation={null} decisions={[]} events={[]} onNavigate={()=>{}}/>)
  expect(markup).toContain('A challenge is pending review')
  expect(markup).toContain('Execution completion and attestation do not establish objective achievement')
 })
 it('restores only matching receipt metadata from replay events',()=>{
  const event=(runId:string,receiptId:string):RuntimeEvent=>({eventId:receiptId,sequence:0,runId,timestamp:'',type:'human_decision',title:'Decision',message:'',status:'completed',metadata:{receipt_id:receiptId,run_id:runId,action:'CHALLENGE'}})
  expect(runDecisionReceipts('RUN-1',[],[event('RUN-1','REC-1'),event('RUN-2','REC-2')]).map(d=>d.receipt_id)).toEqual(['REC-1'])
 })
})
