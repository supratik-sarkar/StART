import {
  APIResponseEnvelope,
  ReviewPresentationExport,
  ReviewerHydrationResponse,
  RunRequest,
  RunStatusResponse,
  SSEEnvelope,
  SystemInfo,
  WebReviewerSubmission,
  ZeroCostAttestation,
} from "../types/start_schema";

const API_BASE = "";

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const resp = await fetch(`${API_BASE}/api/v1/info`);
  const json: APIResponseEnvelope = await resp.json();
  return json.data as SystemInfo;
}

export async function fetchZeroCostAttestation(): Promise<ZeroCostAttestation> {
  const resp = await fetch(`${API_BASE}/api/v1/zero-cost-attestation`);
  const json: APIResponseEnvelope = await resp.json();
  return json.data as ZeroCostAttestation;
}

export async function startAnalyticalRun(
  req: RunRequest
): Promise<{ run_id: string; session_id: string }> {
  const resp = await fetch(`${API_BASE}/api/v1/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  let json: APIResponseEnvelope;
  try {
    json = await resp.json();
  } catch {
    throw new Error(`Server returned HTTP ${resp.status}`);
  }

  if (!resp.ok || !json.success || !json.run_id) {
    if (json.error_code === "TURNSTILE_FAILED") {
      throw new Error("Turnstile security challenge verification failed. Please complete the verification box.");
    } else if (json.error_code === "ENGINE_BUSY") {
      throw new Error("Server compute is currently busy. Your analytical run has been queued.");
    }
    throw new Error(json.error || `Run failed to start (HTTP ${resp.status})`);
  }

  return { run_id: json.run_id, session_id: json.data?.session_id || req.session_id || "" };
}

export async function fetchRunStatus(runId: string, sessionId?: string): Promise<RunStatusResponse> {
  const url = `${API_BASE}/api/v1/runs/${runId}/status${sessionId ? `?session_id=${sessionId}` : ""}`;
  const resp = await fetch(url);
  const json: APIResponseEnvelope = await resp.json();
  return json.data as RunStatusResponse;
}

export async function fetchRunPresentation(runId: string, sessionId?: string): Promise<ReviewPresentationExport> {
  const url = `${API_BASE}/api/v1/runs/${runId}/presentation${sessionId ? `?session_id=${sessionId}` : ""}`;
  const resp = await fetch(url);
  const json: APIResponseEnvelope = await resp.json();
  return json.data?.presentation as ReviewPresentationExport;
}

export async function submitReviewerForHydration(
  runId: string,
  submission: WebReviewerSubmission
): Promise<ReviewerHydrationResponse> {
  const resp = await fetch(`${API_BASE}/api/v1/runs/${runId}/reviewer/hydrate-and-gate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(submission),
  });
  const json: APIResponseEnvelope = await resp.json();
  if (!json.success) {
    throw new Error(json.error || "Failed to hydrate reviewer submission");
  }
  return json.data as ReviewerHydrationResponse;
}

export function subscribeRunEvents(
  runId: string,
  sessionId?: string,
  lastEventId?: string,
  onEvent?: (envelope: SSEEnvelope) => void,
  onComplete?: () => void,
  onError?: (err: any) => void
): () => void {
  const queryParams = new URLSearchParams();
  if (sessionId) queryParams.set("session_id", sessionId);

  const url = `${API_BASE}/api/v1/runs/${runId}/stream?${queryParams.toString()}`;
  const es = new EventSource(url);

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onEvent) onEvent(data as SSEEnvelope);
    } catch (e) {
      console.error("Failed to parse SSE payload", e);
    }
  };

  es.addEventListener("complete", () => {
    if (onComplete) onComplete();
    es.close();
  });

  es.addEventListener("error", (err) => {
    if (onError) onError(err);
    es.close();
  });

  return () => {
    es.close();
  };
}
