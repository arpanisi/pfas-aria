import { Beaker, ChevronRight } from "lucide-react";
import clsx from "clsx";
import type { Hypothesis } from "@/types";

interface Props {
  hypotheses: Hypothesis[];
  selectedId?: string;
  onSelect: (h: Hypothesis) => void;
}

export function HypothesisList({ hypotheses, selectedId, onSelect }: Props) {
  return (
    <div className="card">
      <h3 className="card-title">
        <Beaker size={16} />
        Hypotheses ({hypotheses.length})
      </h3>
      <div className="hypothesis-list">
        {hypotheses.map((h) => (
          <button
            key={h.id}
            className={clsx("hypothesis-item", {
              "hypothesis-item--selected": h.id === selectedId,
              "hypothesis-item--refinement": h.is_refinement,
            })}
            onClick={() => onSelect(h)}
          >
            <div className="hyp-header">
              <span className="hyp-id">{h.hypothesis_id}</span>
              <span className="hyp-round">Round {h.round}</span>
              <span className="hyp-score">
                {(h.priority_score * 100).toFixed(0)}
              </span>
            </div>
            <p className="hyp-description">{h.description}</p>
            <div className="hyp-vars">
              {h.primary_variables.slice(0, 4).map((v) => (
                <span key={v} className="var-chip">
                  {v}
                </span>
              ))}
              {h.primary_variables.length > 4 && (
                <span className="var-chip var-chip--more">
                  +{h.primary_variables.length - 4}
                </span>
              )}
            </div>
            <ChevronRight size={14} className="hyp-arrow" />
          </button>
        ))}
      </div>
    </div>
  );
}
