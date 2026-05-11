// ── API response types ────────────────────────────────────────────────────────

export interface ColumnInfo {
  name: string;
  dtype: string;
  n_unique: number;
  missing_pct: number;
  is_numeric: boolean;
  sample_values: (string | number)[];
}

export interface LegacyRegimeSummary {
  regime_id: number;
  n_rows: number;
  row_indices_sample: string[];
  condition_values_sample: string[];
  /** Input columns with >1 distinct value within this regime's rows. */
  non_constant_input_cols: string[];
  /** Output columns with >1 distinct value within this regime's rows. */
  non_constant_output_cols: string[];
}

export interface RegimeRowCount {
  regime_id: number;
  n_rows: number;
}

export interface LegacySegmentationPreview {
  filename: string;
  n_rows: number;
  n_cols: number;
  n_regimes: number;
  pfoa_col: string;
  pfba_col: string;
  pfbs_col: string;
  n_pfas_input_columns: number;
  input_cols: string[];
  output_cols: string[];
  regimes: LegacyRegimeSummary[];
  /** Row counts for regimes 1–5 (includes zeros for empty patterns). */
  regime_row_counts: RegimeRowCount[];
  warnings: string[];
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
  regime_id?: number | null;
}

export interface AutomatedScreeningIterationRequest {
  filename: string;
  run_name: string;
  regime_id?: number;
  convergence_threshold?: number;
}

export interface AutomatedScreeningIterationResponse {
  hypotheses_tested: number;
  run_id?: string | null;
}

export interface ScreeningGroundedRequest {
  filename: string;
  regime_id: number;
  run_name?: string;
  /** Screening run row to attach grounded hypotheses to (PostgreSQL). */
  run_id?: string | null;
}

export interface ScreeningBundle {
  hypothesis: Hypothesis;
  model_result: ModelResult;
  citations: Citation[];
}

export interface ScreeningGroundedResponse {
  run_name: string;
  filename: string;
  display_title: string;
  dataset_n_rows: number;
  dataset_n_cols: number;
  n_corpus_papers: number;
  regime_id: number;
  regime_n_rows: number;
  bundles: ScreeningBundle[];
  warnings: string[];
  persisted_to_run_id?: string | null;
}

export interface RunStatus {
  run_id: string;
  run_name: string;
  status: string;
  current_round: number;
  final_match_score: number;
  converged: boolean;
  n_rounds_completed: number;
  run_kind: string;
  regime_id: number | null;
  hypotheses_tested: number | null;
  dataset_filename: string | null;
  created_at?: string | null;
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
  /** When set, citation was retrieved for a specific screening hypothesis (e.g. H1). */
  hypothesis_id?: string | null;
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
