import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ConvergencePoint } from "@/types";

interface Props {
  data: ConvergencePoint[];
  threshold: number;
}

export function ConvergenceChart({ data, threshold }: Props) {
  return (
    <div className="card">
      <h3 className="card-title">Convergence History</h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="round"
            label={{ value: "Round", position: "insideBottom", offset: -4 }}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              color: "var(--text)",
            }}
          />
          {/* Threshold line */}
          <Line
            type="monotone"
            dataKey={() => threshold}
            stroke="var(--accent-muted)"
            strokeDasharray="6 3"
            dot={false}
            name="Threshold"
          />
          <Line
            type="monotone"
            dataKey="avg_match_score"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={{ fill: "var(--accent)", r: 4 }}
            name="Match Score"
          />
          <Line
            type="monotone"
            dataKey="best_r_squared"
            stroke="var(--teal)"
            strokeWidth={2}
            dot={{ fill: "var(--teal)", r: 4 }}
            name="Best R²"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
