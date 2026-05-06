import { useNavigate } from "react-router-dom";
import { CheckCircle, Clock, XCircle, Loader } from "lucide-react";
import { useRuns } from "@/hooks/usePipeline";
import type { RunStatus } from "@/types";

const STATUS_ICON: Record<string, React.ReactNode> = {
  converged: <CheckCircle size={16} className="icon-ok" />,
  completed: <CheckCircle size={16} className="icon-muted" />,
  failed: <XCircle size={16} className="icon-err" />,
  running: <Loader size={16} className="icon-spin" />,
  initializing: <Clock size={16} className="icon-muted" />,
};

export function RunHistory() {
  const navigate = useNavigate();
  const { data: runs = [], isLoading } = useRuns();

  return (
    <div className="run-history-page">
      <header className="page-header">
        <h1 className="page-title">Run History</h1>
        <button className="btn-primary" onClick={() => navigate("/upload")}>
          + New Run
        </button>
      </header>

      {isLoading && <p className="loading-text">Loading runs...</p>}

      <div className="run-list">
        {runs.map((run: RunStatus) => (
          <button
            key={run.run_id}
            className="run-card"
            onClick={() => navigate(`/runs/${run.run_id}`)}
          >
            <div className="run-card-header">
              {STATUS_ICON[run.status] ?? <Clock size={16} />}
              <span className="run-name">{run.run_name}</span>
              <span className="run-id">#{run.run_id}</span>
            </div>
            <div className="run-card-meta">
              <span>Round {run.n_rounds_completed}</span>
              <span>
                Match: {(run.final_match_score * 100).toFixed(1)}%
              </span>
              <span className={`status-tag status-tag--${run.status}`}>
                {run.status}
              </span>
            </div>
          </button>
        ))}
        {!isLoading && runs.length === 0 && (
          <div className="empty-state-box">
            <p>No runs yet</p>
            <button
              className="btn-primary"
              onClick={() => navigate("/upload")}
            >
              Start your first run
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
