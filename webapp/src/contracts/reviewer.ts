import type { EvidenceRecord } from './types'

export type ReviewerRuntimeState = 'idle' | 'loading' | 'ready' | 'reviewing' | 'unavailable' | 'error'
export interface ReviewerProgress { state:ReviewerRuntimeState; label:string; percent?:number; downloadedBytes?:number; totalBytes?:number }
export interface ReviewerFinding { findingId:string; title:string; description:string; evidenceIds:string[]; limitations?:string[]; suggestedActions?:string[] }
export interface ReviewerOutput { executiveSummary:string; findings:ReviewerFinding[]; limitations:string[]; evidenceIds:string[]; rawStructuredOutput?:unknown }
export interface ReviewerRuntime {
  readonly runtimeName:string
  initialize(onProgress:(p:ReviewerProgress)=>void):Promise<void>
  review(args:{runId:string;goal:string;evidence:EvidenceRecord[];contextNodeId?:string}, onChunk?:(chunk:string)=>void):Promise<ReviewerOutput>
  dispose():Promise<void>|void
}
