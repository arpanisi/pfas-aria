import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteAllScreeningRuns,
  deleteRun,
  getCitations,
  getConvergence,
  getHypotheses,
  getModelResults,
  getRunStatus,
  getRunSummary,
  listRuns,
  postScreeningGrounded,
  runAutomatedScreeningIteration,
  startRun,
  uploadDataset,
} from "@/api/endpoints";
import type {
  AutomatedScreeningIterationRequest,
  RunConfig,
  ScreeningGroundedRequest,
} from "@/types";

// ── Upload ────────────────────────────────────────────────────────────────────

export function useUploadDataset() {
  return useMutation({
    mutationFn: (file: File) => uploadDataset(file),
  });
}

// ── Run management ────────────────────────────────────────────────────────────

export function useStartRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (config: RunConfig) => startRun(config),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useAutomatedScreeningIteration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: AutomatedScreeningIterationRequest) =>
      runAutomatedScreeningIteration(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => deleteRun(runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useClearAllScreeningRuns() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => deleteAllScreeningRuns(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useScreeningGrounded(
  params: ScreeningGroundedRequest | null,
  enabled: boolean
) {
  return useQuery({
    queryKey: [
      "screeningGrounded",
      params?.filename,
      params?.regime_id,
      params?.run_name,
      params?.run_id,
    ],
    queryFn: () => postScreeningGrounded(params!),
    enabled:
      Boolean(enabled && params) &&
      Boolean(params!.filename) &&
      Number.isFinite(params!.regime_id),
    retry: 0,
  });
}

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: listRuns,
    refetchInterval: 10000,
  });
}

export function useRunStatus(runId: string | null, active: boolean = true) {
  return useQuery({
    queryKey: ["run", runId, "status"],
    queryFn: () => getRunStatus(runId!),
    enabled: !!runId && active,
    refetchInterval: active ? 3000 : false,
  });
}

// ── Results ───────────────────────────────────────────────────────────────────

export function useRunSummary(runId: string | null) {
  return useQuery({
    queryKey: ["run", runId, "summary"],
    queryFn: () => getRunSummary(runId!),
    enabled: !!runId,
  });
}

export function useHypotheses(runId: string | null, round?: number) {
  return useQuery({
    queryKey: ["run", runId, "hypotheses", round],
    queryFn: () => getHypotheses(runId!, round),
    enabled: !!runId,
  });
}

export function useModelResults(runId: string | null) {
  return useQuery({
    queryKey: ["run", runId, "models"],
    queryFn: () => getModelResults(runId!),
    enabled: !!runId,
  });
}

export function useCitations(runId: string | null) {
  return useQuery({
    queryKey: ["run", runId, "citations"],
    queryFn: () => getCitations(runId!),
    enabled: !!runId,
  });
}

export function useConvergence(runId: string | null) {
  return useQuery({
    queryKey: ["run", runId, "convergence"],
    queryFn: () => getConvergence(runId!),
    enabled: !!runId,
  });
}
