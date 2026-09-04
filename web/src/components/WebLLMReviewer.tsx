import React, { useState, useEffect } from "react";
import { Sparkles, Cpu, CheckCircle2, AlertTriangle, ShieldCheck, ArrowRight, Loader2, MessageSquare, Send } from "lucide-react";
import { webLLMService, PINNED_MODEL_ID } from "../services/webllm";
import { submitReviewerForHydration } from "../services/api";
import { HydratedFindingView, ReviewerHydrationResponse, WebReviewerSubmission } from "../types/start_schema";

interface WebLLMReviewerProps {
  runId: string;
  sessionId: string;
  domain: string;
  evidenceRecords: Array<{ evidence_id: string; test_id: string; status: string; metrics: Record<string, any> }>;
  onHydrationComplete?: (resp: ReviewerHydrationResponse) => void;
}

export const WebLLMReviewer: React.FC<WebLLMReviewerProps> = ({
  runId,
  sessionId,
  domain,
  evidenceRecords,
  onHydrationComplete,
}) => {
  const [isSupported, setIsSupported] = useState<boolean | null>(null);
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progressText, setProgressText] = useState("");
  const [progressPercent, setProgressPercent] = useState(0);

  const [isReviewing, setIsReviewing] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const [submission, setSubmission] = useState<WebReviewerSubmission | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);

  const [isHydrating, setIsHydrating] = useState(false);
  const [hydrationResult, setHydrationResult] = useState<ReviewerHydrationResponse | null>(null);

  // Challenge dialogue
  const [challengePrompt, setChallengePrompt] = useState("");
  const [chatLog, setChatLog] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);

  useEffect(() => {
    webLLMService.checkWebGPUSupport().then((sup) => {
      setIsSupported(sup);
      setIsEngineReady(webLLMService.isReady());
    });
  }, []);

  const handleEnableReviewer = async () => {
    setIsLoading(true);
    try {
      await webLLMService.initialize((prog) => {
        setProgressText(prog.text);
        setProgressPercent(prog.progress);
      });
      setIsEngineReady(true);
    } catch (err: any) {
      console.warn("WebLLM initialization notice:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunReview = async () => {
    if (!isEngineReady) return;
    setIsReviewing(true);
    setStreamedText("");
    setSubmission(null);
    setReviewError(null);
    setHydrationResult(null);

    try {
      const result = await webLLMService.generateQualitativeReview(
        runId,
        sessionId,
        domain,
        evidenceRecords,
        (chunk) => {
          setStreamedText((prev) => prev + chunk);
        }
      );
      setSubmission(result);
    } catch (err: any) {
      console.warn("Review generation notice:", err);
      setReviewError(err.message || "Failed to generate structured qualitative review.");
    } finally {
      setIsReviewing(false);
    }
  };

  const handleSubmitForHydration = async () => {
    if (!submission) return;
    setIsHydrating(true);
    try {
      const resp = await submitReviewerForHydration(runId, submission);
      setHydrationResult(resp);
      if (onHydrationComplete) onHydrationComplete(resp);
    } catch (err: any) {
      alert(`Server hydration error: ${err.message || err}`);
    } finally {
      setIsHydrating(false);
    }
  };

  return (
    <div className="bg-white border border-[#E5E5E2] rounded-xl p-4 flex flex-col gap-4 shadow-xs text-left">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-stone-100 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-semibold text-stone-900 tracking-wide">
            Browser AI Reviewer (WebLLM / WebGPU)
          </h3>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px]">
          {isSupported === null ? (
            <span className="text-stone-400">Checking WebGPU...</span>
          ) : isSupported ? (
            <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
              <Cpu className="w-3 h-3" /> WebGPU Ready
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> WebGPU Unavailable
            </span>
          )}
        </div>
      </div>

      {/* Activation / Loading Bar */}
      {!isEngineReady ? (
        <div className="bg-[#FBFBFA] border border-[#E5E5E2] rounded-lg p-3 flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <div className="text-xs text-stone-700">
              <span className="font-semibold">Local Model:</span>{" "}
              <span className="font-mono text-indigo-600">{PINNED_MODEL_ID}</span>
            </div>
            <button
              onClick={handleEnableReviewer}
              disabled={isLoading || !isSupported}
              className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-md transition-colors flex items-center gap-1.5 disabled:opacity-50 cursor-pointer shadow-xs"
            >
              {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>Initialize Local AI</span>
            </button>
          </div>
          {isLoading && (
            <div className="flex flex-col gap-1.5 mt-1">
              <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-indigo-600 h-full transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(5, progressPercent))}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[10px] font-mono text-stone-500">
                <span>{progressText || "Downloading weights into browser cache..."}</span>
                <span>{progressPercent.toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between bg-emerald-50/60 border border-emerald-200 rounded-lg p-2.5">
          <div className="flex items-center gap-2 text-xs font-mono text-emerald-800">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>Local Engine Ready: {PINNED_MODEL_ID}</span>
          </div>
          <button
            onClick={handleRunReview}
            disabled={isReviewing || evidenceRecords.length === 0}
            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-md transition-colors flex items-center gap-1.5 disabled:opacity-50 cursor-pointer shadow-xs"
          >
            {isReviewing && <Loader2 className="w-3 h-3 animate-spin" />}
            <span>Synthesize Qualitative Review</span>
          </button>
        </div>
      )}

      {/* Streamed Output */}
      {isReviewing && (
        <div className="bg-[#FBFBFA] border border-[#E5E5E2] rounded-lg p-3 font-mono text-xs text-stone-800 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
          {streamedText || "Generating structured assessment over permitted evidence records..."}
        </div>
      )}

      {/* Parse Error Notice (No fake findings fabricated) */}
      {reviewError && (
        <div className="bg-rose-50 border border-rose-200 rounded-lg p-3 text-xs text-rose-800 flex items-start justify-between gap-3">
          <div className="flex flex-col gap-1">
            <span className="font-semibold">Review Parse Notice:</span>
            <span className="font-mono text-[11px]">{reviewError}</span>
            <span className="text-[11px] text-rose-600">
              Deterministic StART results remain verified and available. You may retry structured inference.
            </span>
          </div>
          <button
            onClick={handleRunReview}
            className="px-3 py-1 bg-rose-600 hover:bg-rose-700 text-white font-medium rounded text-xs shrink-0 cursor-pointer shadow-xs"
          >
            Retry Review
          </button>
        </div>
      )}

      {/* Structured Submission View */}
      {submission && !hydrationResult && (
        <div className="flex flex-col gap-2.5 bg-[#FBFBFA] border border-[#E5E5E2] rounded-lg p-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold text-stone-900">
              Generated Findings ({(submission.findings || []).length})
            </h4>
            <button
              onClick={handleSubmitForHydration}
              disabled={isHydrating}
              className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-md flex items-center gap-1.5 disabled:opacity-50 cursor-pointer shadow-xs"
            >
              {isHydrating && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>Submit for Server-Side Gating</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(submission.findings || []).map((f, i) => (
              <div key={i} className="p-2.5 rounded bg-white border border-[#E5E5E2] text-xs shadow-2xs">
                <div className="flex items-center justify-between font-semibold text-stone-900 mb-1">
                  <span>{f.title}</span>
                  <span className="text-[10px] font-mono font-medium text-stone-500">{f.severity}</span>
                </div>
                <p className="text-stone-600 mb-1.5 leading-relaxed">{f.description}</p>
                <div className="flex flex-wrap gap-1 font-mono text-[10px]">
                  {(f.evidence_refs || []).map((r, ri) => (
                    <span
                      key={ri}
                      className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200"
                    >
                      [{r.evidence_id}] {r.metric_name}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Server Hydration & OPA Governance Result */}
      {hydrationResult && (
        <div className="flex flex-col gap-2 bg-emerald-50/50 border border-emerald-200 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <span className="text-xs font-semibold text-stone-900">
                Server Hydration & OPA Policy Complete
              </span>
            </div>
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300 font-medium">
                Disposition: {hydrationResult.governance_disposition}
              </span>
              <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 border border-indigo-300 font-medium">
                OPA: {hydrationResult.opa_policy_decision}
              </span>
            </div>
          </div>
          <div className="text-[11px] font-mono text-stone-500 truncate">
            Attestation Seal Merkle Root: {hydrationResult.attestation_seal_merkle_root}
          </div>
        </div>
      )}
    </div>
  );
};
