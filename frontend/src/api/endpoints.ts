import { apiClient } from "./client";
import type {
  Citation,
  ConvergencePoint,
  CorpusStats,
  DatasetPreview,
  Hypothesis,
  ModelResult,
  RunConfig,
  RunStatus,
  RunSummary,
} from "@/types";

// ── Pipeline ──────────────────────────────────────────────────────────────────

export const uploadDataset = async (file: File): Promise<DatasetPreview> => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post("/pipeline/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const startRun = async (
  config: RunConfig
): Promise<{ run_id: string; status: string }> => {
  const { data } = await apiClient.post("/pipeline/run", config);
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
  file: File
): Promise<{ filename: string; status: string }> => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post("/corpus/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};
