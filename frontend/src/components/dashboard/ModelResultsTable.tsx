import { CheckCircle, XCircle } from "lucide-react";
import type { ModelResult } from "@/types";

interface Props {
  results: ModelResult[];
}

export function ModelResultsTable({ results }: Props) {
  return (
    <div className="card">
      <h3 className="card-title">Model Results</h3>
      <div className="table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>R²</th>
              <th>Adj R²</th>
              <th>N</th>
              <th>Match</th>
              <th>Valid</th>
              <th>Top Variables</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.id}>
                <td>
                  <span className="model-badge">{r.model_type}</span>
                </td>
                <td className="num">{r.r_squared.toFixed(3)}</td>
                <td className="num">{r.adj_r_squared.toFixed(3)}</td>
                <td className="num">{r.n_observations}</td>
                <td>
                  <div className="match-bar-wrap">
                    <div
                      className="match-bar"
                      style={{ width: `${r.match_score * 100}%` }}
                    />
                    <span className="match-val">
                      {(r.match_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td>
                  {r.validation_passed ? (
                    <CheckCircle size={16} className="icon-ok" />
                  ) : (
                    <XCircle size={16} className="icon-err" />
                  )}
                </td>
                <td>
                  <div className="sig-vars">
                    {r.significant_variables.slice(0, 3).map((v) => (
                      <span key={v} className="var-chip">
                        {v}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
