import React, { useState, useEffect } from "react";
import { Sparkles, Cpu, CheckCircle2, AlertTriangle, ShieldCheck, ArrowRight, Loader2 } from "lucide-react";
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

  const [isHydrating, setIsHydrating] = useState(false);
  const [hydrationResult, setHydrationResult] = useState<ReviewerHydrationResponse | null>(null);

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
      alert(`WebLLM initialization error: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunReview = async () => {
    if (!isEngineReady) return;
    setIsReviewing(true);
    setStreamedText("");
    setSubmission(null);
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
      alert(`Review generation error: ${err.message || err}`);
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
    <div className="bg-card border border-border rounded-lg p-4 flex flex-col gap-4 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/60 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
            Browser WebLLM Reviewer (WebGPU Client Inference)
          </h3>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px]">
          {isSupported === null ? (
            <span className="text-muted-foreground">Checking WebGPU...</span>
          ) : isSupported ? (
            <span className="px-2 py-0.5 rounded bg-green-950/60 text-green-400 border border-green-800/60 flex items-center gap-1">
              <Cpu className="w-3 h-3" /> WebGPU Supported
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded bg-amber-950/60 text-amber-400 border border-amber-800/60 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> WebGPU Unavailable
            </span>
          )}
        </div>
      </div>

      {/* Model Activation Bar */}
      {!isEngineReady ? (
        <div className="bg-background/80 border border-border rounded-lg p-3 flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <div className="text-xs text-foreground">
              <span className="font-semibold">Selected Local Model:</span>{" "}
              <span className="font-mono text-primary">{PINNED_MODEL_ID}</span> (~1.1 GB WebAssembly/WebGPU)
            </div>
            <button
              onClick={handleEnableReviewer}
              disabled={isLoading || !isSupported}
              className="px-3 py-1.5 bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-semibold rounded transition-colors flex items-center gap-1.5 disabled:opacity-50"
            >
              {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>Enable Browser AI Reviewer</span>
            </button>
          </div>
          {isLoading && (
            <div className="flex flex-col gap-1 mt-1">
              <div className="w-full bg-secondary h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-primary h-full transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(5, progressPercent))}%` }}
                />
              </div>
              <span className="text-[11px] font-mono text-muted-foreground">{progressText || "Downloading weights into browser cache..."}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-between bg-green-950/20 border border-green-900/40 rounded-lg p-2.5">
          <div className="flex items-center gap-2 text-xs font-mono text-green-400">
            <CheckCircle2 className="w-4 h-4" />
            <span>Local Engine Ready: {PINNED_MODEL_ID}</span>
          </div>
          <button
            onClick={handleRunReview}
            disabled={isReviewing || evidenceRecords.length === 0}
            className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded transition-colors flex items-center gap-1.5 disabled:opacity-50"
          >
            {isReviewing && <Loader2 className="w-3 h-3 animate-spin" />}
            <span>Execute Qualitative Review</span>
          </button>
        </div>
      )}

      {/* Streamed Review Output */}
      {isReviewing && (
        <div className="bg-background border border-border rounded-lg p-3 font-mono text-xs text-foreground/90 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
          {streamedText || "Generating structured assessment over permitted evidence records..."}
        </div>
      )}

      {/* Structured Submission View */}
      {submission && !hydrationResult && (
        <div className="flex flex-col gap-2.5 bg-background border border-border rounded-lg p-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold text-foreground">Generated Findings ({(submission.findings || []).length})</h4>
            <button
              onClick={handleSubmitForHydration}
              disabled={isHydrating}
              className="px-3 py-1 bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-semibold rounded flex items-center gap-1.5 disabled:opacity-50"
            >
              {isHydrating && <Loader2 className="w-3 h-3 animate-spin" />}
              <span>Submit for Server-Side Gating</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(submission.findings || []).map((f, i) => (
              <div key={i} className="p-2 rounded bg-card border border-border/80 text-xs">
                <div className="flex items-center justify-between font-bold text-primary mb-1">
                  <span>{f.title}</span>
                  <span className="text-[10px] font-mono text-muted-foreground">{f.severity}</span>
                </div>
                <p className="text-muted-foreground mb-1.5">{f.description}</p>
                <div className="flex flex-wrap gap-1 font-mono text-[10px]">
                  {(f.evidence_refs || []).map((r, ri) => (
                    <span key={ri} className="px-1.5 py-0.5 rounded bg-blue-950/40 text-blue-400 border border-blue-900/50">
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
        <div className="flex flex-col gap-2 bg-gradient-to-r from-green-950/20 to-purple-950/20 border border-green-800/60 rounded-lg p-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-green-400" />
              <span className="text-xs font-bold text-foreground">
                Server-Side Hydration & OPA Governance Complete
              </span>
            </div>
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="px-2 py-0.5 rounded bg-green-950/80 text-green-400 border border-green-700">
                Disposition: {hydrationResult.governance_disposition}
              </span>
              <span className="px-2 py-0.5 rounded bg-purple-950/80 text-purple-400 border border-purple-700">
                OPA: {hydrationResult.opa_policy_decision}
              </span>
            </div>
          </div>
          <div className="text-[11px] font-mono text-muted-foreground truncate">
            Merkle Attestation Root: {hydrationResult.attestation_seal_merkle_root}
          </div>
        </div>
      )}
    </div>
  );
};
