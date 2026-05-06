import { useNavigate } from "react-router-dom";
import { useRuns } from "@/hooks/usePipeline";

export function RunHistory() {
  const navigate = useNavigate();
  const { data: runs = [], isLoading } = useRuns();

  return (
    <div className="run-history-page">
      <div className="page-header">
        <h1 className="page-title">Run History</h1>
        <button className="btn-primary" onClick={() => navigate("/upload")}>+ New Run</button>
      </div>
      {isLoading && <p style={{ color: "var(--text-muted)", padding: "24px 0" }}>Loading...</p>}
      <div className="run-list">
        {runs.map((r) => (
          <div key={r.run_id} className="run-card" onClick={() => navigate(`/runs/${r.run_id}`)}>
            <div className="run-card-top">
              <span className="run-card-name">{r.run_name}</span>
              <span className="run-card-id">#{r.run_id}</span>
            </div>
            <div className="run-card-meta">
              <span>Round {r.n_rounds_completed}</span>
              <span>Match: {(r.final_match_score * 100).toFixed(1)}%</span>
              <span className={`status-tag ${r.status}`}>{r.status}</span>
            </div>
          </div>
        ))}
        {!isLoading && runs.length === 0 && (
          <div className="empty-box">
            <p>No runs yet</p>
            <button className="btn-primary" onClick={() => navigate("/upload")}>Start your first run</button>
          </div>
        )}
      </div>
    </div>
  );
}
