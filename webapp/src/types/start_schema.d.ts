// ─────────────────────────────────────────────────────────────────────────────
// AUTOMATICALLY GENERATED FROM start.web.schemas (PYTHON PYDANTIC SOURCE)
// DO NOT EDIT MANUALLY — Run: python scripts/export_web_schemas.py
// ─────────────────────────────────────────────────────────────────────────────

export interface SystemInfo {
  start_version?: string;
  start_schema_version?: string;
  backend_build_version?: string;
  git_sha?: string | null;
  compute_runtime?: string;
  max_concurrency?: number;
  engine_status?: "READY" | "BUSY" | "MAINTENANCE";
  supported_domains?: string[];
  synthetic_profiles?: string[];
}

export interface APIResponseEnvelope {
  success?: boolean;
  schema_version?: string;
  run_id?: string | null;
  timestamp?: number;
  data?: Record<string, any>;
  error?: string | null;
  error_code?: string | null;
}

export interface SSEEnvelope {
  event_id?: string;
  sequence?: number;
  run_id: string;
  timestamp?: number;
  event_type?: string;
  schema_version?: string;
  source_agent?: string;
  target_agent?: string;
  stage?: string;
  action?: string;
  status?: string;
  latency_ms?: number;
  evidence_refs?: string[];
  artifact_refs?: string[];
  policy_decision?: string;
  payload?: Record<string, any>;
  phase?: string;
  step?: number;
  completed?: number;
  total?: number;
  percent?: number;
  elapsed_seconds?: number;
  estimated_remaining_seconds?: number | null;
  message?: string;
}

export interface RunRequest {
  domain?: "predictive" | "deep_learning" | "market";
  mode?: "deterministic" | "llm";
  materiality?: "low" | "medium" | "high";
  lifecycle?: "pre_implementation" | "validation" | "annual_review" | "monitoring";
  synthetic_profile?: string;
  synthetic_profile_version?: string;
  seed?: number;
  turnstile_token?: string | null;
  session_id?: string;
  workflow?: string;
  workflowId?: string | null;
  contextId?: string | null;
  goal?: string | null;
  sourceEvidenceId?: string | null;
  parentRunId?: string | null;
  parameters?: Record<string, any>;
  parent_run_id?: string | null;
  intervention?: string | null;
}

export interface RunStatusResponse {
  run_id: string;
  session_id: string;
  status: "CONFIGURING" | "VALIDATING" | "QUEUED" | "INITIALIZING" | "RUNNING" | "PARTIAL" | "COMPLETED" | "RECOVERABLE_FAILURE" | "FAILED" | "BUSY";
  domain: string;
  synthetic_profile: string;
  created_at: number;
  completed_at?: number | null;
  event_count?: number;
  evidence_count?: number;
  artifact_count?: number;
  error_message?: string | null;
  error_code?: string | null;
  queue_position?: number;
  phase?: string;
  step?: number;
  completed?: number;
  total?: number;
  percent?: number;
  elapsed_seconds?: number;
}

export interface MetricRowView {
  test_id: string;
  metric: string;
  value: any;
  unit?: string;
  status?: string;
  evidence_id?: string;
  artifact_id?: string | null;
}

export interface PresentationBlockView {
  block_id: string;
  title: string;
  domain: string;
  rows?: MetricRowView[];
  summary?: Record<string, any>;
  artifacts?: Record<string, any>[];
}

export interface ReviewPresentationExport {
  run_id: string;
  mode: string;
  domains: string[];
  materiality: string;
  lifecycle: string;
  governance_disposition: string;
  attestation_seal_merkle_root: string;
  blocks?: Record<string, PresentationBlockView>;
  orchestration_events?: Record<string, any>[];
}

export interface LogicalArtifactMetadata {
  artifact_id: string;
  run_id: string;
  title: string;
  artifact_type: "svg" | "json" | "html" | "pdf" | "table" | "csv";
  evidence_ids?: string[];
  description?: string;
  size_bytes?: number;
  sha256?: string;
}

export interface QualitativeFinding {
  finding_id?: string;
  severity?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  title: string;
  description: string;
  evidence_refs?: EvidenceMetricRef[];
  recommendation?: string;
}

export interface WebReviewerSubmission {
  run_id: string;
  session_id: string;
  model_name?: string;
  executive_summary?: string;
  findings?: QualitativeFinding[];
  limitations?: string[];
  suggested_actions?: string[];
}

export interface HydratedFindingView {
  finding_id: string;
  severity: string;
  title: string;
  description: string;
  grounded: boolean;
  evidence_refs?: Record<string, any>[];
  recommendation?: string;
}

export interface ReviewerHydrationResponse {
  run_id: string;
  schema_version?: string;
  model_name: string;
  is_grounded: boolean;
  hydrated_findings?: HydratedFindingView[];
  opa_policy_decision?: "ALLOW" | "WARN" | "BLOCK" | "DENY";
  opa_reasons?: string[];
  governance_disposition?: "ACCEPT" | "CONDITIONAL_ACCEPT" | "REJECT";
  attestation_seal_merkle_root?: string;
  attestation_timestamp?: number;
}
