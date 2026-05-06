import { Activity } from "lucide-react";
import clsx from "clsx";
import type { RunStep } from "@/types";

const STEPS: { key: RunStep; label: string }[] = [
  { key: "ingesting", label: "Ingest Data" },
  { key: "building_rag", label: "Build RAG" },
  { key: "analyzing_data", label: "Analyze" },
  { key: "generating_hypotheses", label: "Hypotheses" },
  { key: "modeling", label: "Model" },
  { key: "grounding", label: "Ground" },
];

interface Props {
  currentStage: string;
  round: number;
  matchScore: number;
  connected: boolean;
}

export function LiveStatus({ currentStage, round, matchScore, connected }: Props) {
  const currentIdx = STEPS.findIndex((s) => s.key === currentStage);

  return (
    <div className="card live-status">
      <div className="live-header">
        <h3 className="card-title">
          <Activity size={16} />
          Live Status
        </h3>
        <div className={clsx("ws-dot", { "ws-dot--on": connected })} />
      </div>

      <div className="round-badge">Round {round}</div>

      <div className="progress-steps">
        {STEPS.map((step, i) => (
          <div
            key={step.key}
            className={clsx("progress-step", {
              "progress-step--done": i < currentIdx,
              "progress-step--active": i === currentIdx,
              "progress-step--pending": i > currentIdx,
            })}
          >
            <div className="step-dot" />
            <span className="step-label">{step.label}</span>
          </div>
        ))}
      </div>

      <div className="match-score-display">
        <span className="match-label">Match Score</span>
        <span className="match-value">{(matchScore * 100).toFixed(1)}%</span>
        <div className="match-track">
          <div
            className="match-fill"
            style={{ width: `${Math.min(matchScore * 100, 100)}%` }}
          />
          <div className="match-threshold" style={{ left: "75%" }} />
        </div>
      </div>
    </div>
  );
}
