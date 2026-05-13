import type { MouseEvent } from "react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import { useClearAllScreeningRuns, useDeleteRun, useRuns } from "@/hooks/usePipeline";
import type { RunStatus } from "@/types";

function formatWhen(iso: string | null | undefined) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

/** Backend full-table screening segment uses regime key 1 (see dataset_screening_layout). */
const DEFAULT_SCREENING_REGIME_ID = 1;

function openRun(navigate: ReturnType<typeof useNavigate>, r: RunStatus) {
  if (r.run_kind === "screening" && r.dataset_filename) {
    const regimeId = r.regime_id != null ? r.regime_id : DEFAULT_SCREENING_REGIME_ID;
    const q = new URLSearchParams({
      filename: r.dataset_filename,
      regimeId: String(regimeId),
      runId: r.run_id,
    });
    navigate(`/runs/stats?${q.toString()}`, { state: { runName: r.run_name } });
    return;
  }
  navigate(`/runs/${r.run_id}`);
}

export function RunHistory() {
  const navigate = useNavigate();
  const { data: runs = [], isLoading } = useRuns();
  const deleteMutation = useDeleteRun();
  const clearScreeningMutation = useClearAllScreeningRuns();

  const handleClearScreening = () => {
    const ok = window.confirm(
      "Delete all screening (hypothesis test) runs from history? Full pipeline runs are kept. This cannot be undone.",
    );
    if (!ok) return;
    clearScreeningMutation.mutate(undefined, {
      onSuccess: (res) => toast.success(`Removed ${res.deleted} screening run(s).`),
      onError: () => toast.error("Could not clear screening runs."),
    });
  };

  const handleDelete = (e: MouseEvent, r: RunStatus) => {
    e.stopPropagation();
    const ok = window.confirm(`Delete run “${r.run_name}” (#${r.run_id})? This cannot be undone.`);
    if (!ok) return;
    deleteMutation.mutate(r.run_id, {
      onSuccess: () => toast.success("Run deleted"),
      onError: () => toast.error("Could not delete run"),
    });
  };

  return (
    <div className="run-history-page">
      <div className="page-header">
        <h1 className="page-title">Run history</h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn-secondary"
            disabled={clearScreeningMutation.isPending}
            onClick={handleClearScreening}
          >
            Clear hypothesis tests
          </button>
          <button type="button" className="btn-primary" onClick={() => navigate("/upload")}>
            + New run
          </button>
        </div>
      </div>
      {isLoading && <p style={{ color: "var(--text-muted)", padding: "24px 0" }}>Loading…</p>}
      <div className="run-list" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {runs.map((r) => (
          <div
            key={r.run_id}
            className="run-card"
            role="button"
            tabIndex={0}
            onClick={() => openRun(navigate, r)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                openRun(navigate, r);
              }
            }}
          >
            <div className="run-card-row">
              <div className="run-card-main">
                <div className="run-card-top">
                  <span className={`run-kind-pill ${r.run_kind === "screening" ? "screening" : ""}`}>
                    {r.run_kind === "screening" ? "Screening" : "Pipeline"}
                  </span>
                  <span className="run-card-name">{r.run_name}</span>
                  <span className="run-card-id">#{r.run_id}</span>
                </div>
                <div className="run-card-meta">
                  <span title="When recorded">{formatWhen(r.created_at)}</span>
                  {r.dataset_filename && (
                    <span title="Dataset file">File: {r.dataset_filename}</span>
                  )}
                  <span>
                    Regime:{" "}
                    {r.regime_id != null ? (
                      <strong>{r.regime_id}</strong>
                    ) : (
                      <span style={{ opacity: 0.7 }}>—</span>
                    )}
                  </span>
                  <span>
                    Hypotheses tested:{" "}
                    <strong>
                      {r.hypotheses_tested != null ? r.hypotheses_tested.toLocaleString() : "—"}
                    </strong>
                  </span>
                  <span>Rounds: {r.n_rounds_completed}</span>
                  <span>Match: {(r.final_match_score * 100).toFixed(1)}%</span>
                  <span className={`status-tag ${r.status}`}>{r.status}</span>
                </div>
              </div>
              <div className="run-card-actions">
                <button
                  type="button"
                  className="run-card-delete"
                  title="Delete run"
                  aria-label={`Delete run ${r.run_id}`}
                  disabled={deleteMutation.isPending}
                  onClick={(e) => handleDelete(e, r)}
                >
                  ×
                </button>
              </div>
            </div>
          </div>
        ))}
        {!isLoading && runs.length === 0 && (
          <div className="empty-box">
            <p>No runs yet. Run hypothesis screening on New Run, or start a full pipeline run.</p>
            <button type="button" className="btn-primary" onClick={() => navigate("/upload")}>
              Start your first run
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
