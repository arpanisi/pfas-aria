import { useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useScreeningStats } from "@/hooks/usePipeline";
import type { Hypothesis, ModelResult } from "@/types";

type EffectView = {
  variable: string;
  coefficient: number;
  pValue?: number;
  significant: boolean;
  width: number;
  sigLabel: string;
};

function significanceLabel(p?: number) {
  if (p === undefined || Number.isNaN(p)) return "—";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

function formatCoef(value: number) {
  const sign = value >= 0 ? "+" : "−";
  return `${sign}${Math.abs(value).toFixed(4)}`;
}

function buildEffects(model: ModelResult | null): EffectView[] {
  if (!model) return [];
  const values = Object.values(model.coefficients ?? {}).map((v) => Math.abs(v));
  const max = Math.max(...values, 0.01);
  return Object.entries(model.coefficients ?? {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)
    .map(([variable, coefficient]) => {
      const pValue = model.p_values?.[variable];
      return {
        variable,
        coefficient,
        pValue,
        significant:
          model.significant_variables?.includes(variable) ||
          (typeof pValue === "number" && pValue < 0.05),
        width: Math.max(8, Math.round((Math.abs(coefficient) / max) * 48)),
        sigLabel: significanceLabel(pValue),
      };
    });
}

function modelForHypothesis(hypothesis: Hypothesis | null, modelResults: ModelResult[]) {
  if (!hypothesis) return modelResults[0] ?? null;
  return (
    modelResults.find((m) => m.hypothesis_id === hypothesis.hypothesis_id) ??
    modelResults.find((m) => m.hypothesis_id === hypothesis.id) ??
    modelResults[0] ??
    null
  );
}

export function ScreeningStats() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();

  const filename = searchParams.get("filename") ?? "";
  const regimeIdRaw = searchParams.get("regimeId");
  const regimeId = regimeIdRaw != null && regimeIdRaw !== "" ? Number(regimeIdRaw) : NaN;
  const runIdParam = searchParams.get("runId")?.trim() || undefined;
  const runName =
    (location.state as { runName?: string } | null)?.runName?.trim() || "Screening run";

  const request = useMemo(() => {
    if (!filename || Number.isNaN(regimeId)) return null;
    return { filename, regime_id: regimeId, run_name: runName, run_id: runIdParam ?? null };
  }, [filename, regimeId, runName, runIdParam]);

  const query = useScreeningStats(request, Boolean(request));

  const [selectedHypId, setSelectedHypId] = useState<string | null>(null);

  const bundles = useMemo(() => query.data?.bundles ?? [], [query.data?.bundles]);
  const hypotheses = useMemo(() => bundles.map((b) => b.hypothesis), [bundles]);
  const modelResults = useMemo(() => bundles.map((b) => b.model_result), [bundles]);

  const selectedHyp =
    hypotheses.find((h) => h.id === selectedHypId) ??
    hypotheses.find((h) => h.hypothesis_id === selectedHypId) ??
    hypotheses[0] ??
    null;

  const selectedModel = modelForHypothesis(selectedHyp, modelResults);
  const effects = useMemo(() => buildEffects(selectedModel), [selectedModel]);

  const bestR2 = modelResults.length ? Math.max(...modelResults.map((m) => m.r_squared)) : 0;

  const groundingUrl = useMemo(() => {
    if (!filename || Number.isNaN(regimeId)) return null;
    const q = new URLSearchParams({ filename, regimeId: String(regimeId) });
    if (runIdParam) q.set("runId", runIdParam);
    return `/runs/screening?${q.toString()}`;
  }, [filename, regimeId, runIdParam]);

  if (!filename || Number.isNaN(regimeId)) {
    return (
      <div className="results-shell">
        <div className="section-card" style={{ maxWidth: 560, margin: "48px auto" }}>
          <h1 className="page-title">Screening results</h1>
          <p style={{ color: "var(--text-muted)", marginBottom: 16 }}>
            Open this view from <strong>New Run</strong> after screening, or pass{" "}
            <code>?filename=…&amp;regimeId=…</code> in the URL.
          </p>
          <button type="button" className="btn-primary" onClick={() => navigate("/upload")}>
            Go to New Run
          </button>
        </div>
      </div>
    );
  }

  if (query.isLoading) {
    return (
      <div className="results-shell">
        <p style={{ color: "var(--text-muted)", padding: 48, maxWidth: 520 }}>
          Running statistical screening… fitting models across all input subsets and outputs.
        </p>
      </div>
    );
  }

  if (query.isError) {
    const msg =
      typeof (query.error as { message?: string })?.message === "string"
        ? (query.error as { message: string }).message
        : "Request failed";
    return (
      <div className="results-shell">
        <div className="section-card" style={{ maxWidth: 560, margin: "48px auto" }}>
          <h1 className="page-title">Could not load results</h1>
          <p style={{ color: "var(--text-muted)" }}>{msg}</p>
          <button
            type="button"
            className="btn-primary"
            style={{ marginTop: 16 }}
            onClick={() => navigate("/upload")}
          >
            Back to New Run
          </button>
        </div>
      </div>
    );
  }

  const d = query.data;

  return (
    <div className="results-shell">
      <div className="results-main">
        <section className="hero-card">
          <div>
            <div className="eyebrow">Statistical screening · top fits by R²</div>
            <h1>{d?.display_title ?? "Statistical Screening Results"}</h1>
            <p>
              Hypotheses below are the strongest statistical associations in{" "}
              <strong>regime {d?.regime_id ?? regimeId}</strong>, ranked by in-sample R².
              No corpus or literature grounding required for this view.
            </p>
          </div>
          <div className="hero-metrics">
            <div className="metric-card">
              <span>Observations</span>
              <strong>{d?.dataset_n_rows?.toLocaleString() ?? "—"}</strong>
              <small>{d?.dataset_n_cols ?? "—"} columns</small>
            </div>
            <div className="metric-card">
              <span>Top hypotheses</span>
              <strong>{hypotheses.length}</strong>
              <small>Regime {d?.regime_id ?? regimeId} · n = {d?.regime_n_rows?.toLocaleString() ?? "—"}</small>
            </div>
            <div className="metric-card accent-metric">
              <span>Best R²</span>
              <strong>{bestR2 > 0 ? bestR2.toFixed(3) : "—"}</strong>
              <small>Top in-sample fit in this regime</small>
            </div>
          </div>
        </section>

        {d?.warnings?.length ? (
          <div className="section-card" style={{ marginBottom: 16 }}>
            {d.warnings.map((w) => (
              <p key={w} style={{ color: "var(--text-muted)", margin: "0 0 8px" }}>
                {w}
              </p>
            ))}
          </div>
        ) : null}

        <section className="content-grid">
          <div className="left-stack">
            <div className="section-card">
              <div className="section-head">
                <div>
                  <div className="eyebrow">ranked by R²</div>
                  <h2>Top hypotheses</h2>
                </div>
              </div>
              <div className="hypothesis-grid">
                {hypotheses.map((hypothesis) => {
                  const model = modelForHypothesis(hypothesis, modelResults);
                  const selected =
                    selectedHyp?.id === hypothesis.id ||
                    (!selectedHyp && hypothesis === hypotheses[0]);
                  return (
                    <button
                      key={hypothesis.id}
                      type="button"
                      className={`hypothesis-card ${selected ? "selected" : ""}`}
                      onClick={() => setSelectedHypId(hypothesis.id)}
                    >
                      <div className="hypothesis-topline">
                        <span>{hypothesis.hypothesis_id}</span>
                        <b style={{ marginLeft: "auto" }}>R² {model?.r_squared?.toFixed(3) ?? "—"}</b>
                      </div>
                      <p>{hypothesis.description}</p>
                      <div className="chip-row">
                        {hypothesis.primary_variables.slice(0, 5).map((v) => (
                          <span key={v}>{v}</span>
                        ))}
                      </div>
                      <div className="hypothesis-evidence">
                        <div>
                          <small>adj R²</small>
                          <strong>{model?.adj_r_squared?.toFixed(3) ?? "—"}</strong>
                        </div>
                        <div>
                          <small>n obs</small>
                          <strong>{model?.n_observations ?? "—"}</strong>
                        </div>
                        <div>
                          <small>sig. vars</small>
                          <strong>{model?.significant_variables?.length ?? 0}</strong>
                        </div>
                      </div>
                    </button>
                  );
                })}
                {!hypotheses.length && (
                  <div className="empty-inline">
                    No hypotheses met the R² threshold for this regime.
                  </div>
                )}
              </div>
            </div>

            {selectedHyp && selectedModel && (
              <div className="section-card">
                <div className="section-head">
                  <div>
                    <div className="eyebrow">selected result</div>
                    <h2>{selectedHyp.hypothesis_id}: parameter effects</h2>
                  </div>
                  <div className="model-pill">
                    {selectedModel.model_type.replace(/_/g, " ")}
                  </div>
                </div>
                <div className="evidence-layout">
                  <div className="interpretation-card">
                    <h3>Fit summary</h3>
                    <div className="validation-list">
                      <span>
                        n = {selectedModel.n_observations} observations
                      </span>
                      <span>
                        R² {selectedModel.r_squared.toFixed(4)} · adj R²{" "}
                        {selectedModel.adj_r_squared.toFixed(4)}
                      </span>
                      <span>
                        {selectedModel.significant_variables.length} significant variable
                        {selectedModel.significant_variables.length !== 1 ? "s" : ""} (p &lt; 0.05)
                      </span>
                    </div>
                    {selectedModel.significant_variables.length > 0 && (
                      <div style={{ marginTop: 12 }}>
                        <small style={{ color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                          Significant predictors
                        </small>
                        <div className="chip-row">
                          {selectedModel.significant_variables.map((v) => (
                            <span key={v} style={{ background: "var(--accent-muted, #e8f0fe)", color: "var(--accent)" }}>
                              {v}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="effect-card">
                    <h3>Coefficients (sorted by magnitude)</h3>
                    <div className="effect-list refined">
                      {effects.map((effect) => (
                        <div key={effect.variable} className="effect-row refined">
                          <div className="effect-name" title={effect.variable}>
                            {effect.variable}
                          </div>
                          <div className="effect-axis">
                            <i />
                            <b
                              className={effect.coefficient >= 0 ? "positive" : "negative"}
                              style={{ width: `${effect.width}%` }}
                            />
                          </div>
                          <div
                            className={
                              effect.coefficient >= 0
                                ? "effect-value positive"
                                : "effect-value negative"
                            }
                          >
                            {formatCoef(effect.coefficient)}
                          </div>
                          <div
                            className={`effect-sig ${effect.sigLabel === "ns" ? "muted" : ""}`}
                          >
                            {effect.sigLabel}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <aside className="grounding-panel">
            <div className="section-head compact">
              <div>
                <div className="eyebrow">Next step</div>
                <h2>Literature grounding</h2>
              </div>
            </div>
            <p className="grounding-note">
              These results show pure statistical fit. Upload PDF papers to the corpus to
              rank hypotheses by resemblance to published literature.
            </p>
            <div style={{ marginTop: 16 }}>
              <button
                type="button"
                className="btn-primary"
                style={{ width: "100%" }}
                onClick={() => groundingUrl && navigate(groundingUrl, { state: { runName } })}
              >
                Run literature grounding
              </button>
              <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
                Requires at least 3 PDF papers uploaded to the corpus.
              </p>
            </div>
            <div style={{ marginTop: 24, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8 }}>
                Upload research papers to get corpus-grounded rankings for these hypotheses.
              </p>
              <button
                type="button"
                className="btn-secondary"
                style={{ width: "100%" }}
                onClick={() => navigate("/corpus")}
              >
                Manage corpus
              </button>
            </div>
          </aside>
        </section>

        <div style={{ marginTop: 24 }}>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => navigate("/upload")}
          >
            Back to New Run
          </button>
        </div>
      </div>
    </div>
  );
}
