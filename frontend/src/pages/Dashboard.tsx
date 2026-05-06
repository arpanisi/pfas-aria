import { useState } from "react";
import { useParams } from "react-router-dom";
import { useWebSocket } from "@/hooks/useWebSocket";
import {
  useCitations, useConvergence, useHypotheses,
  useModelResults, useRunStatus, useRunSummary,
} from "@/hooks/usePipeline";
import type { Hypothesis, ModelResult } from "@/types";

const EFFECTS_MOCK: Record<string, { v: string; e: string; w: number; pos: boolean; sig: string }[]> = {
  default: [
    { v: "uv_intensity", e: "+0.81", w: 62, pos: true, sig: "***" },
    { v: "ph", e: "−0.56", w: 44, pos: false, sig: "**" },
    { v: "temperature_c", e: "+0.34", w: 28, pos: true, sig: "*" },
    { v: "uv × ph", e: "−0.21", w: 18, pos: false, sig: "ns" },
  ],
};

export function Dashboard() {
  const { runId } = useParams<{ runId: string }>();
  const [selectedHypId, setSelectedHypId] = useState<string | null>(null);
  const [activeRound, setActiveRound] = useState<number | undefined>(undefined);

  const { lastMessage } = useWebSocket(runId ?? null);
  const { data: summary } = useRunSummary(runId ?? null);
  const { data: status } = useRunStatus(runId ?? null, true);
  const { data: hypotheses = [] } = useHypotheses(runId ?? null, activeRound);
  const { data: modelResults = [] } = useModelResults(runId ?? null);
  const { data: citations = [] } = useCitations(runId ?? null);

  const matchScore = lastMessage?.match_score ?? status?.final_match_score ?? 0;
  const currentRound = lastMessage?.round ?? status?.current_round ?? 0;
  const runStatus = status?.status ?? "loading";
  const maxRound = summary?.n_rounds ?? 0;
  const rounds = Array.from({ length: maxRound }, (_, i) => i + 1);

  const selectedHyp = hypotheses.find((h) => h.id === selectedHypId) ?? hypotheses[0] ?? null;
  const effects = EFFECTS_MOCK["default"];

  const SOURCE_LABEL: Record<string, string> = {
    corpus: "Your Papers", arxiv: "arXiv", semantic_scholar: "Semantic Scholar",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", height: "100vh", overflow: "hidden" }}>
      {/* CENTER */}
      <div className="center-panel">
        {/* Status bar */}
        <div className="card status-bar">
          <div>
            <div className="run-name">{summary?.run_name ?? "Run"}</div>
            <div className="run-meta">
              Round {currentRound} of {summary?.n_rounds ?? "?"} ·
              outcome: {summary?.outcome_variable ?? "—"} ·
              {summary?.n_hypotheses ?? 0} hypotheses
            </div>
          </div>
          <div className="status-right">
            <div className="match-display">
              <span className="match-label">match</span>
              <span className="match-value">{(matchScore * 100).toFixed(0)}%</span>
              <div className="match-track">
                <div className="match-fill" style={{ width: `${Math.min(matchScore * 100, 100)}%` }} />
                <div className="match-thresh" />
              </div>
            </div>
            <div className={`status-badge badge-${runStatus}`}>{runStatus}</div>
          </div>
        </div>

        {/* Regimes placeholder */}
        <div className="card">
          <div className="card-title">Detected Regimes</div>
          <div className="regime-grid">
            {[
              { name: "Regime A", n: 124, uv: "8–15 mW/cm²", ph: "6.8–7.4", rate: "0.31 /hr" },
              { name: "Regime B", n: 116, uv: "22–38 mW/cm²", ph: "7.8–8.6", rate: "0.71 /hr" },
            ].map((r) => (
              <div key={r.name} className="regime-card">
                <div className="reg-header">
                  <span className="reg-name">{r.name}</span>
                  <span className="reg-n">n = {r.n}</span>
                </div>
                <div className="reg-stat">
                  <div className="reg-row"><span className="reg-key">UV intensity</span><span className="reg-val">{r.uv}</span></div>
                  <div className="reg-row"><span className="reg-key">pH</span><span className="reg-val">{r.ph}</span></div>
                  <div className="reg-row"><span className="reg-key">Avg rate</span><span className="reg-val">{r.rate}</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hypotheses */}
        <div className="card">
          <div className="card-title">
            Hypotheses
            {maxRound > 0 && (
              <span style={{ marginLeft: 12, display: "inline-flex", gap: 4 }}>
                <button
                  className={`col-btn ${activeRound === undefined ? "feature active" : ""}`}
                  onClick={() => setActiveRound(undefined)}
                >All</button>
                {rounds.map((r) => (
                  <button
                    key={r}
                    className={`col-btn ${activeRound === r ? "feature active" : ""}`}
                    onClick={() => setActiveRound(r)}
                  >R{r}</button>
                ))}
              </span>
            )}
          </div>
          <div className="hyp-list">
            {hypotheses.map((h) => (
              <div
                key={h.id}
                className={`hyp-card ${selectedHypId === h.id ? "selected" : ""}`}
                onClick={() => setSelectedHypId(h.id)}
              >
                <div className="hyp-header">
                  <span className="hyp-id">{h.hypothesis_id}</span>
                  <span className="hyp-round">Round {h.round}</span>
                  <span className={`hyp-status ${h.is_refinement ? "hs-testing" : "hs-done"}`}>
                    {h.is_refinement ? "testing" : "tested"}
                  </span>
                </div>
                <div className="hyp-desc">{h.description}</div>
                <div className="chip-list">
                  {h.primary_variables.slice(0, 4).map((v) => (
                    <span key={v} className="chip">{v}</span>
                  ))}
                  {h.primary_variables.length > 4 && (
                    <span className="chip">+{h.primary_variables.length - 4}</span>
                  )}
                </div>
              </div>
            ))}
            {hypotheses.length === 0 && (
              <p style={{ color: "var(--text-dim)", fontSize: 12, padding: "12px 0" }}>
                No hypotheses yet — pipeline is running
              </p>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div className="right-panel">
        <div className="panel-title">
          Parameter Effects{selectedHyp ? ` — ${selectedHyp.hypothesis_id}` : ""}
        </div>

        <div className="effect-list">
          {effects.map((e) => (
            <div key={e.v} className="effect-row">
              <div className="eff-var">{e.v}</div>
              <div className="eff-bar-wrap">
                <div
                  className={`eff-bar ${e.pos ? "bar-pos" : "bar-neg"}`}
                  style={{ width: `${e.w / 2}%` }}
                />
              </div>
              <div className="eff-val" style={{ color: e.pos ? "var(--accent)" : "var(--red)" }}>
                {e.e}
              </div>
              <div className={`eff-sig ${e.sig === "ns" ? "ns" : ""}`}>{e.sig}</div>
            </div>
          ))}
        </div>

        <div className="sig-legend" style={{ marginTop: 4, marginBottom: 4 }}>
          <span>*** p&lt;0.001</span>
          <span>** p&lt;0.01</span>
          <span>* p&lt;0.05</span>
        </div>

        <div className="panel-sep" />
        <div className="panel-title">Literature Support ({citations.length})</div>

        <div className="cit-list">
          {citations.slice(0, 8).map((c) => (
            <div key={c.id} className="cit-card">
              <div className="cit-header">
                <span className={`cit-src src-${c.source === "semantic_scholar" ? "s2" : c.source}`}>
                  {SOURCE_LABEL[c.source] ?? c.source}
                </span>
                {c.year && <span className="cit-year">{c.year}</span>}
                <span className="cit-score">{(c.similarity_score * 100).toFixed(0)}%</span>
              </div>
              <div className="cit-title">{c.title}</div>
            </div>
          ))}
          {citations.length === 0 && (
            <p style={{ color: "var(--text-dim)", fontSize: 11 }}>No citations yet</p>
          )}
        </div>
      </div>
    </div>
  );
}
