import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteAllScreeningRuns,
  deleteRun,
  getCitations,
  getConvergence,
  getGroundingProgress,
  getSavedScreeningGrounded,
  getHypotheses,
  getModelResults,
  getRunStatus,
  getRunSummary,
  listRuns,
  postScreeningGrounded,
  postScreeningStats,
  runAutomatedScreeningIteration,
  startScreeningGrounded,
  uploadDataset,
} from "@/api/endpoints";
import type {
  AutomatedScreeningIterationRequest,
  GroundingJobProgress,
  ScreeningGroundedRequest,
  ScreeningGroundedResponse,
  ScreeningStatsRequest,
} from "@/types";

// ── Upload ────────────────────────────────────────────────────────────────────

export function useUploadDataset() {
  return useMutation({
    mutationFn: (file: File) => uploadDataset(file),
  });
}

// ── Run management ────────────────────────────────────────────────────────────

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

export function useGroundingProgress(
  params: ScreeningGroundedRequest | null,
  enabled: boolean
): {
  data: ScreeningGroundedResponse | null;
  pct: number;
  stage: string;
  etaSeconds: number | null;
  isLoading: boolean;
  isError: boolean;
  error: string | null;
} {
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<GroundingJobProgress | null>(null);
  const [savedData, setSavedData] = useState<ScreeningGroundedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stable key to detect param changes and reset
  const paramKey = params
    ? `${params.filename}|${params.regime_id}|${params.run_id ?? ""}`
    : null;
  const prevParamKey = useRef<string | null>(null);

  useEffect(() => {
    if (paramKey !== prevParamKey.current) {
      // Params changed — reset everything
      setJobId(null);
      setProgress(null);
      setSavedData(null);
      setError(null);
      startedRef.current = false;
      prevParamKey.current = paramKey;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [paramKey]);

  // Start job
  useEffect(() => {
    if (!enabled || !params || startedRef.current) return;
    startedRef.current = true;

    const startJob = () => {
      startScreeningGrounded(params)
        .then(({ job_id }) => setJobId(job_id))
        .catch((err: unknown) => {
          const msg =
            (err as { response?: { data?: { detail?: { message?: string } | string } } })
              ?.response?.data?.detail as string | { message?: string } | undefined;
          setError(
            typeof msg === "object" && msg?.message
              ? msg.message
              : typeof msg === "string"
              ? msg
              : String(err)
          );
        });
    };

    if (params.run_id) {
      getSavedScreeningGrounded(params.run_id)
        .then((data) => setSavedData(data))
        .catch((err: unknown) => {
          const statusCode = (err as { response?: { status?: number } })?.response?.status;
          if (statusCode === 404) {
            startJob();
            return;
          }
          const msg =
            (err as { response?: { data?: { detail?: { message?: string } | string } } })
              ?.response?.data?.detail as string | { message?: string } | undefined;
          setError(
            typeof msg === "object" && msg?.message
              ? msg.message
              : typeof msg === "string"
              ? msg
              : String(err)
          );
        });
      return;
    }

    startJob();
  }, [enabled, params]);

  // Poll progress
  useEffect(() => {
    if (!jobId || progress?.done) return;
    const iv = setInterval(() => {
      getGroundingProgress(jobId)
        .then((p) => {
          setProgress(p);
          if (p.done) clearInterval(iv);
        })
        .catch((err: unknown) => {
          setError(String(err));
          clearInterval(iv);
        });
    }, 1500);
    intervalRef.current = iv;
    return () => clearInterval(iv);
  }, [jobId, progress?.done]);

  const data =
    savedData ?? (progress?.done && !progress.error && progress.result ? progress.result : null);
  const isLoading = Boolean(enabled && params && !data && !progress?.done && !error);
  const isError = Boolean(error || (progress?.done && progress.error));
  const resolvedError = error ?? (progress?.done ? (progress.error ?? null) : null);

  return {
    data,
    pct: savedData ? 100 : progress?.pct ?? 0,
    stage: savedData ? "Done" : progress?.stage ?? (isLoading ? "Starting…" : ""),
    etaSeconds: savedData ? null : progress?.eta_seconds ?? null,
    isLoading,
    isError,
    error: resolvedError,
  };
}

export function useScreeningStats(
  params: ScreeningStatsRequest | null,
  enabled: boolean
) {
  return useQuery({
    queryKey: ["screeningStats", params?.filename, params?.regime_id, params?.run_name, params?.run_id],
    queryFn: () => postScreeningStats(params!),
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
