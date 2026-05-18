import { apiClient } from "./client";
import type {
  AutomatedScreeningIterationRequest,
  AutomatedScreeningIterationResponse,
  Citation,
  ConvergencePoint,
  CorpusStats,
  DatasetPreview,
  GroundingJobProgress,
  GroundingJobStart,
  Hypothesis,
  LegacySegmentationPreview,
  ModelResult,
  RunStatus,
  RunSummary,
  ScreeningGroundedRequest,
  ScreeningGroundedResponse,
  ScreeningStatsRequest,
  ScreeningStatsResponse,
} from "@/types";

// ── Pipeline ──────────────────────────────────────────────────────────────────

export const uploadDataset = async (file: File): Promise<DatasetPreview> => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post("/pipeline/upload", formData);
  return data;
};

export const getLegacySegmentationPreview = async (
  filename: string
): Promise<LegacySegmentationPreview> => {
  const { data } = await apiClient.get("/pipeline/legacy-segmentation-preview", {
    params: { filename },
  });
  return data;
};

export const runAutomatedScreeningIteration = async (
  body: AutomatedScreeningIterationRequest
): Promise<AutomatedScreeningIterationResponse> => {
  const { data } = await apiClient.post(
    "/pipeline/automated-screening-iteration",
    body,
    { timeout: 120_000 }
  );
  return data;
};

export const postScreeningGrounded = async (
  body: ScreeningGroundedRequest
): Promise<ScreeningGroundedResponse> => {
  const { data } = await apiClient.post("/pipeline/screening-grounded", body, {
    // Sync screening + many RAG calls + LLM rationales can exceed 2 minutes on real datasets.
    timeout: 600_000,
  });
  return data;
};

export const startScreeningGrounded = async (
  body: ScreeningGroundedRequest
): Promise<GroundingJobStart> => {
  const { data } = await apiClient.post("/pipeline/screening-grounded/start", body);
  return data;
};

export const getGroundingProgress = async (jobId: string): Promise<GroundingJobProgress> => {
  const { data } = await apiClient.get(`/pipeline/screening-grounded/progress/${encodeURIComponent(jobId)}`);
  return data;
};

export const postScreeningStats = async (
  body: ScreeningStatsRequest
): Promise<ScreeningStatsResponse> => {
  const { data } = await apiClient.post("/pipeline/screening-stats", body, {
    timeout: 120_000,
  });
  return data;
};

export const getRunStatus = async (runId: string): Promise<RunStatus> => {
  const { data } = await apiClient.get(`/pipeline/status/${runId}`);
  return data;
};

export const listRuns = async (): Promise<RunStatus[]> => {
  const { data } = await apiClient.get("/pipeline/runs");
  return data;
};

export const deleteRun = async (runId: string): Promise<void> => {
  await apiClient.delete(`/pipeline/runs/${encodeURIComponent(runId)}`);
};

export const deleteAllScreeningRuns = async (): Promise<{ deleted: number }> => {
  const { data } = await apiClient.delete("/pipeline/runs/screening/all");
  return data;
};

// ── Results ───────────────────────────────────────────────────────────────────

export const getRunSummary = async (runId: string): Promise<RunSummary> => {
  const { data } = await apiClient.get(`/results/${runId}/summary`);
  return data;
};

export const getHypotheses = async (
  runId: string,
  round?: number
): Promise<Hypothesis[]> => {
  const params = round !== undefined ? { round_number: round } : {};
  const { data } = await apiClient.get(`/results/${runId}/hypotheses`, {
    params,
  });
  return data;
};

export const getModelResults = async (
  runId: string
): Promise<ModelResult[]> => {
  const { data } = await apiClient.get(`/results/${runId}/models`);
  return data;
};

export const getCitations = async (runId: string): Promise<Citation[]> => {
  const { data } = await apiClient.get(`/results/${runId}/citations`);
  return data;
};

export const getConvergence = async (
  runId: string
): Promise<ConvergencePoint[]> => {
  const { data } = await apiClient.get(`/results/${runId}/convergence`);
  return data;
};

// ── Corpus ────────────────────────────────────────────────────────────────────

export const getCorpusStats = async (): Promise<CorpusStats> => {
  const { data } = await apiClient.get("/corpus/stats");
  return data;
};

export const uploadPaper = async (
  file: File,
  onProgress?: (pct: number) => void,
): Promise<{ filename: string; status: string }> => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post("/corpus/upload", formData, {
    timeout: 0,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
};

export async function deletePaper(paperId: string): Promise<void> {
  await apiClient.delete(`/corpus/${encodeURIComponent(paperId)}`);
}

export async function clearCorpus(): Promise<{
  status: string;
  deleted_papers: number;
  deleted_chunks: number;
}> {
  const { data } = await apiClient.delete("/corpus");
  return data;
}

export async function inferDomainContext(
  colNames: string[]
): Promise<{ domain_context: string; n_papers: number; corpus_fingerprint: string }> {
  const { data } = await apiClient.post("/corpus/domain-context", {
    col_names: colNames,
  });
  return data;
}
