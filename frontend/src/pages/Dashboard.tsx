import { useState } from "react";
import { useParams } from "react-router-dom";
import { useWebSocket } from "@/hooks/useWebSocket";
import {
  useCitations,
  useConvergence,
  useHypotheses,
  useModelResults,
  useRunStatus,
  useRunSummary,
} from "@/hooks/usePipeline";
import { ConvergenceChart } from "@/components/dashboard/ConvergenceChart";
import { HypothesisList } from "@/components/dashboard/HypothesisList";
import { ModelResultsTable } from "@/components/dashboard/ModelResultsTable";
import { CitationPanel } from "@/components/dashboard/CitationPanel";
import { LiveStatus } from "@/components/dashboard/LiveStatus";
import type { Hypothesis } from "@/types";

export function Dashboard() {
  const { runId } = useParams<{ runId: string }>();
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [activeRound, setActiveRound] = useState<number | undefined>(undefined);

  const isActive = true;
  const { lastMessage, connected } = useWebSocket(runId ?? null);

  const { data: summary } = useRunSummary(runId ?? null);
  const { data: status } = useRunStatus(runId ?? null, isActive);
  const { data: hypotheses = [] } = useHypotheses(runId ?? null, activeRound);
  const { data: modelResults = [] } = useModelResults(runId ?? null);
  const { data: citations = [] } = useCitations(runId ?? null);
  const { data: convergence = [] } = useConvergence(runId ?? null);

  const currentStage = lastMessage?.stage ?? status?.status ?? "initializing";
  const currentRound = lastMessage?.round ?? status?.current_round ?? 0;
  const matchScore = lastMessage?.match_score ?? status?.final_match_score ?? 0;

  const maxRound = summary?.n_rounds ?? 0;
  const rounds = Array.from({ length: maxRound }, (_, i) => i + 1);

  return (
    <div className="dashboard">
      <header className="page-header">
        <div>
          <h1 className="page-title">{summary?.run_name ?? "Run"}</h1>
          <p className="page-subtitle">
            {summary?.outcome_variable && `Outcome: ${summary.outcome_variable}`}
            {summary?.n_hypotheses !== undefined &&
              ` · ${summary.n_hypotheses} hypotheses`}
          </p>
        </div>
        <div className="status-pill" data-status={status?.status}>
          {status?.status ?? "loading"}
        </div>
      </header>

      <div className="dashboard-grid">
        {/* Left column */}
        <div className="col-left">
          <LiveStatus
            currentStage={currentStage}
            round={currentRound}
            matchScore={matchScore}
            connected={connected}
          />

          {convergence.length > 0 && (
            <ConvergenceChart data={convergence} threshold={0.75} />
          )}

          {/* Round filter */}
          {maxRound > 0 && (
            <div className="round-filter">
              <button
                className={activeRound === undefined ? "round-btn--active" : "round-btn"}
                onClick={() => setActiveRound(undefined)}
              >
                All
              </button>
              {rounds.map((r) => (
                <button
                  key={r}
                  className={activeRound === r ? "round-btn--active" : "round-btn"}
                  onClick={() => setActiveRound(r)}
                >
                  R{r}
                </button>
              ))}
            </div>
          )}

          <HypothesisList
            hypotheses={hypotheses}
            selectedId={selectedHypothesis?.id}
            onSelect={setSelectedHypothesis}
          />
        </div>

        {/* Right column */}
        <div className="col-right">
          <ModelResultsTable results={modelResults} />
          <CitationPanel citations={citations} />
        </div>
      </div>
    </div>
  );
}
