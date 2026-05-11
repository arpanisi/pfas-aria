// ── API response types ────────────────────────────────────────────────────────

export interface ColumnInfo {
  name: string;
  dtype: string;
  n_unique: number;
  missing_pct: number;
  is_numeric: boolean;
  sample_values: (string | number)[];
}

export interface DatasetPreview {
  filename: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnInfo[];
  /** 0-based Excel row used as column headers; omitted for CSV/TSV */
  excel_header_row?: number | null;
}

export interface RunConfig {
  run_name: string;
  filename: string;
  outcome_variable: string;
  feature_columns: string[];
  exclude_columns: string[];
  max_rounds: number;
  convergence_threshold: number;
  hypotheses_per_round: number;
  strict_validation: boolean;
}

export interface RunStatus {
  run_id: string;
  run_name: string;
  status: string;
  current_round: number;
  final_match_score: number;
  converged: boolean;
  n_rounds_completed: number;
}

export interface Hypothesis {
  id: string;
  hypothesis_id: string;
  round: number;
  description: string;
  rationale: string | null;
  primary_variables: string[];
  model_family: string;
  priority_score: number;
  is_refinement: boolean;
}

export interface ModelResult {
  id: string;
  hypothesis_id: string;
  model_type: string;
  r_squared: number;
  adj_r_squared: number;
  n_observations: number;
  coefficients: Record<string, number>;
  p_values: Record<string, number>;
  significant_variables: string[];
  match_score: number;
  validation_passed: boolean;
}

export interface Citation {
  id: string;
  source: string;
  title: string;
  url: string | null;
  year: string | null;
  similarity_score: number;
  variable: string | null;
}

export interface ConvergencePoint {
  round: number;
  avg_match_score: number;
  best_r_squared: number;
  n_hypotheses: number;
  n_passed: number;
}

export interface RunSummary {
  run_id: string;
  run_name: string;
  status: string;
  n_rounds: number;
  final_match_score: number;
  converged: boolean;
  outcome_variable: string | null;
  selected_features: string[];
  n_hypotheses: number;
}

export interface PaperOut {
  id: string;
  filename: string;
  title: string | null;
  n_chunks: number;
  n_tokens: number;
  embedding_model: string | null;
  source: string;
}

export interface CorpusStats {
  n_papers: number;
  n_chunks_total: number;
  n_tokens_total: number;
  papers: PaperOut[];
}

// ── WebSocket message types ───────────────────────────────────────────────────

export interface WSStatusMessage {
  type: "status" | "keepalive";
  run_id: string;
  stage: string;
  round: number;
  match_score: number;
  details: Record<string, unknown>;
}

// ── UI state types ────────────────────────────────────────────────────────────

export type UploadStep = "upload" | "configure" | "running" | "results";

export type RunStep =
  | "initializing"
  | "ingesting"
  | "building_rag"
  | "analyzing_data"
  | "generating_hypotheses"
  | "modeling"
  | "grounding"
  | "converged"
  | "completed"
  | "failed";
