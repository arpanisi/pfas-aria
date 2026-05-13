import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useGroundingProgress } from "@/hooks/usePipeline";
import type { Citation, Hypothesis, ModelResult, ScreeningBundle } from "@/types";

type EffectView = {
  variable: string;
  coefficient: number;
  pValue?: number;
  significant: boolean;
  width: number;
  sigLabel: string;
};

type GroundingCard = Citation & {
  normalizedSource: string;
  scorePct: number;
  reasonTags: string[];
};

const SOURCE_LABEL: Record<string, string> = {
  corpus: "Uploaded",
  uploaded: "Uploaded",
  arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar",
  s2: "Semantic Scholar",
  crossref: "Crossref",
};

function sigmoidScore(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function significanceLabel(p?: number) {
  if (p === undefined || Number.isNaN(p)) return "—";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

function formatCoef(value: number) {
  const sign = value >= 0 ? "+" : "−";
  return `${sign}${Math.abs(value).toFixed(2)}`;
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

function buildEffects(model: ModelResult | null): EffectView[] {
  if (!model) return [];
  const values = Object.values(model.coefficients ?? {}).map((v) => Math.abs(v));
  const max = Math.max(...values, 0.01);
  return Object.entries(model.coefficients ?? {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8)
    .map(([variable, coefficient]) => {
      const pValue = model.p_values?.[variable];
      return {
        variable,
        coefficient,
        pValue,
        significant:
          model.significant_variables?.includes(variable) ||
          (typeof pValue === "number" && pValue < 0.05),
        width: Math.max(10, Math.round((Math.abs(coefficient) / max) * 48)),
        sigLabel: significanceLabel(pValue),
      };
    });
}

function normalizeSource(source: string) {
  const src = source?.toLowerCase?.() ?? "unknown";
  if (src.includes("semantic")) return "semantic_scholar";
  if (src.includes("arxiv")) return "arxiv";
  if (src.includes("crossref")) return "crossref";
  if (src.includes("upload") || src.includes("corpus") || src.includes("paper")) return "corpus";
  return src;
}

function reasonTags(citation: Citation, selected: Hypothesis | null) {
  const title = citation.title.toLowerCase();
  const tags = new Set<string>();
  if (citation.variable) tags.add(citation.variable);
  selected?.primary_variables?.slice(0, 2).forEach((v) => tags.add(v));
  if (title.includes("plasma")) tags.add("plasma reactor");
  if (title.includes("electrochemical")) tags.add("electrochemical");
  if (title.includes("pfca") || title.includes("pfoa")) tags.add("PFCA/PFOA");
  if (title.includes("pfas")) tags.add("PFAS");
  return Array.from(tags).slice(0, 4);
}

function buildGrounding(citations: Citation[], selected: Hypothesis | null): GroundingCard[] {
  return [...citations]
    .sort((a, b) => (b.similarity_score ?? 0) - (a.similarity_score ?? 0))
    .slice(0, 8)
    .map((citation) => {
      const normalizedSource = normalizeSource(citation.source);
      return {
        ...citation,
        normalizedSource,
        scorePct: Math.round((citation.similarity_score ?? 0) * 100),
        reasonTags: reasonTags(citation, selected),
      };
    });
}

function topValidation(model: ModelResult | null, matchScore: number) {
  if (!model) return ["Awaiting model evidence", "Grounding not yet available"];
  const lines = [
    `${model.model_type.replace(/_/g, " ")} · n=${model.n_observations}`,
    `R² ${model.r_squared.toFixed(3)} · adjusted R² ${model.adj_r_squared.toFixed(3)}`,
    model.validation_passed ? "Validation passed" : "Validation requires review",
    `Literature resemblance ${sigmoidScore(matchScore || model.match_score)}`,
  ];
  return lines;
}


export function ScreeningResults() {
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
    return {
      filename,
      regime_id: regimeId,
      run_name: runName,
      ...(runIdParam ? { run_id: runIdParam } : {}),
    };
  }, [filename, regimeId, runName, runIdParam]);

  const enabled = Boolean(request);
  const query = useGroundingProgress(request, enabled);

  const [selectedHypId, setSelectedHypId] = useState<string | null>(null);

  const bundles = query.data?.bundles ?? [];
  const hypotheses = useMemo(() => bundles.map((b) => b.hypothesis), [bundles]);
  const modelResults = useMemo(() => bundles.map((b) => b.model_result), [bundles]);

  useEffect(() => {
    if (query.error?.includes("CORPUS_TOO_SMALL")) {
      navigate("/corpus?needPapers=1", { replace: true });
    }
  }, [query.error, navigate]);

  useEffect(() => {
    if (!hypotheses.length) return;
    setSelectedHypId((prev) => {
      if (prev && hypotheses.some((h) => h.id === prev || h.hypothesis_id === prev)) return prev;
      return hypotheses[0].id;
    });
  }, [hypotheses]);

  const selectedHyp =
    hypotheses.find((h) => h.id === selectedHypId) ??
    hypotheses.find((h) => h.hypothesis_id === selectedHypId) ??
    hypotheses[0] ??
    null;

  const selectedBundle = useMemo((): ScreeningBundle | null => {
    if (!selectedHyp) return null;
    return (
      bundles.find((b) => b.hypothesis.id === selectedHyp.id) ??
      bundles.find((b) => b.hypothesis.hypothesis_id === selectedHyp.hypothesis_id) ??
      null
    );
  }, [bundles, selectedHyp]);

  const selectedModel = modelForHypothesis(selectedHyp, modelResults);
  const effects = useMemo(() => buildEffects(selectedModel), [selectedModel]);
  const citationsForHyp = selectedBundle?.citations ?? [];
  const grounding = useMemo(
    () => buildGrounding(citationsForHyp, selectedHyp),
    [citationsForHyp, selectedHyp],
  );

  const matchScore = selectedModel?.match_score ?? 0;
  const bestR2 = modelResults.length ? Math.max(...modelResults.map((m) => m.r_squared)) : 0;

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
    const { pct, stage, etaSeconds } = query;
    const etaLabel =
      etaSeconds != null && etaSeconds > 0
        ? etaSeconds >= 60
          ? `~${Math.ceil(etaSeconds / 60)} min remaining`
          : `~${etaSeconds}s remaining`
        : null;
    return (
      <div className="results-shell">
        <div className="section-card" style={{ maxWidth: 520, margin: "64px auto", padding: "36px 32px" }}>
          <h2 style={{ marginBottom: 24, fontSize: 18 }}>Building grounded results…</h2>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, alignItems: "baseline" }}>
            <span style={{ color: "var(--text-muted)", fontSize: 13 }}>{stage || "Starting…"}</span>
            <span style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
              {etaLabel && (
                <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{etaLabel}</span>
              )}
              <span style={{ fontWeight: 600, fontSize: 13 }}>{pct}%</span>
            </span>
          </div>
          <div style={{ background: "var(--border)", borderRadius: 8, height: 10, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${pct}%`,
              background: "var(--accent, #4f8ef7)",
              borderRadius: 8,
              transition: "width 0.6s ease",
            }} />
          </div>
        </div>
      </div>
    );
  }

  if (query.isError && !query.error?.includes("CORPUS_TOO_SMALL")) {
    return (
      <div className="results-shell">
        <div className="section-card" style={{ maxWidth: 560, margin: "48px auto" }}>
          <h1 className="page-title">Could not load results</h1>
          <p style={{ color: "var(--text-muted)" }}>{query.error ?? "Request failed"}</p>
          <button type="button" className="btn-primary" style={{ marginTop: 16 }} onClick={() => navigate("/upload")}>
            Back to New Run
          </button>
        </div>
      </div>
    );
  }

  if (query.isError && query.error?.includes("CORPUS_TOO_SMALL")) {
    return null;
  }

  const d = query.data;

  return (
    <div className="results-shell">
      <div className="results-main">
        <section className="hero-card">
          <div>
            <div className="eyebrow">Hypothesis → panel test → result → mechanistic interpretation → ranked literature support</div>
            <h1>{d?.display_title ?? "Literature Grounding of Discovered Signals"}</h1>
            <p>
              Results are limited to <strong>regime {d?.regime_id ?? regimeId}</strong> only — the regime you chose on
              Configure. Hypotheses below passed automated screening and were ranked by fit and resemblance to your
              uploaded corpus.
            </p>
            {d?.persisted_to_run_id && (
              <p style={{ marginTop: 12, fontSize: "13px", color: "var(--text-muted)" }}>
                Grounded hypotheses saved to PostgreSQL for run <strong>#{d.persisted_to_run_id}</strong>.
              </p>
            )}
            {!runIdParam && query.data && (
              <p style={{ marginTop: 12, fontSize: "13px", color: "var(--text-muted)" }}>
                Pass a screening <code>runId</code> in the URL (from New Run after screening) to store these results in
                run history.
              </p>
            )}
          </div>
          <div className="hero-metrics">
            <div className="metric-card">
              <span>Uploaded observations</span>
              <strong>{d?.dataset_n_rows?.toLocaleString() ?? "—"}</strong>
              <small>{d?.dataset_n_cols ?? "—"} columns</small>
            </div>
            <div className="metric-card">
              <span>Hypotheses rendered</span>
              <strong>{hypotheses.length}</strong>
              <small>Regime {d?.regime_id ?? regimeId} · n = {d?.regime_n_rows?.toLocaleString() ?? "—"}</small>
            </div>
            <div className="metric-card">
              <span>Best panel fit</span>
              <strong>{bestR2 > 0 ? bestR2.toFixed(3) : "—"}</strong>
              <small>Top in-sample R² in this slice</small>
            </div>
            <div className="metric-card accent-metric">
              <span>Grounding corpus</span>
              <strong>{d?.n_corpus_papers ?? "—"}</strong>
              <small>uploaded papers (min. 3)</small>
            </div>
          </div>
        </section>

        <section className="workflow-strip">
          {["ingest", "hypothesize", "model", "validate", "ground"].map((step) => (
            <div
              key={step}
              className={`workflow-step ${["hypothesize", "model", "validate", "ground"].includes(step) ? "active" : ""}`}
            >
              <span>{step}</span>
            </div>
          ))}
          <div className="status-badge badge-completed">static</div>
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
                  <div className="eyebrow">single regime</div>
                  <h2>Detected experimental system</h2>
                </div>
              </div>
              <div className="hypothesis-grid" style={{ gridTemplateColumns: "1fr" }}>
                <div className="hypothesis-card selected" style={{ cursor: "default" }}>
                  <div className="hypothesis-topline">
                    <span>Regime {d?.regime_id}</span>
                    <em>{d?.regime_n_rows?.toLocaleString() ?? "—"} rows</em>
                  </div>
                  <p>
                    Screening and literature ranking use only this regime slice from{" "}
                    <strong>{d?.filename}</strong>.
                  </p>
                </div>
              </div>
            </div>

            <div className="section-card">
              <div className="section-head">
                <div>
                  <div className="eyebrow">tested claims</div>
                  <h2>Hypotheses (literature-ranked)</h2>
                </div>
              </div>
              <div className="hypothesis-grid">
                {hypotheses.map((hypothesis) => {
                  const model = modelForHypothesis(hypothesis, modelResults);
                  const selected = selectedHyp?.id === hypothesis.id;
                  return (
                    <button
                      key={hypothesis.id}
                      type="button"
                      className={`hypothesis-card ${selected ? "selected" : ""}`}
                      onClick={() => setSelectedHypId(hypothesis.id)}
                    >
                      <div className="hypothesis-topline">
                        <span>{hypothesis.hypothesis_id}</span>
                        <em>Round {hypothesis.round}</em>
                        <b>{hypothesis.is_refinement ? "refinement" : "screened"}</b>
                      </div>
                      <p>{hypothesis.description}</p>
                      <div className="chip-row">
                        {hypothesis.primary_variables.slice(0, 5).map((variable) => (
                          <span key={variable}>{variable}</span>
                        ))}
                      </div>
                      <div className="hypothesis-evidence">
                        <div>
                          <small>Model</small>
                          <strong>{model?.model_type?.replace(/_/g, " ") ?? hypothesis.model_family}</strong>
                        </div>
                        <div>
                          <small>R²</small>
                          <strong>{model?.r_squared?.toFixed(3) ?? "—"}</strong>
                        </div>
                        <div>
                          <small>Match</small>
                          <strong>{sigmoidScore(model?.match_score ?? hypothesis.priority_score)}</strong>
                        </div>
                      </div>
                    </button>
                  );
                })}
                {!hypotheses.length && (
                  <div className="empty-inline">No hypotheses met the screening and literature filters for this regime.</div>
                )}
              </div>
            </div>

            <div className="section-card">
              <div className="section-head">
                <div>
                  <div className="eyebrow">selected result</div>
                  <h2>{selectedHyp?.hypothesis_id ?? "Hypothesis"}: statistical evidence</h2>
                </div>
                <div className="model-pill">{selectedModel?.model_type?.replace(/_/g, " ") ?? "model pending"}</div>
              </div>

              <div className="evidence-layout">
                <div className="interpretation-card">
                  <h3>Interpretation</h3>
                  <p>
                    {selectedHyp?.rationale ??
                      "Select a hypothesis to see how the automated fit aligns with retrieved corpus passages."}
                  </p>
                  <div className="validation-list">
                    {topValidation(selectedModel, matchScore).map((line) => (
                      <span key={line}>{line}</span>
                    ))}
                  </div>
                </div>

                <div className="effect-card">
                  <h3>Parameter effects</h3>
                  <div className="effect-list refined">
                    {effects.map((effect) => (
                      <div key={effect.variable} className="effect-row refined">
                        <div className="effect-name">{effect.variable}</div>
                        <div className="effect-axis">
                          <i />
                          <b
                            className={effect.coefficient >= 0 ? "positive" : "negative"}
                            style={{ width: `${effect.width}%` }}
                          />
                        </div>
                        <div className={effect.coefficient >= 0 ? "effect-value positive" : "effect-value negative"}>
                          {formatCoef(effect.coefficient)}
                        </div>
                        <div className={`effect-sig ${effect.sigLabel === "ns" ? "muted" : ""}`}>{effect.sigLabel}</div>
                      </div>
                    ))}
                    {!effects.length && (
                      <div className="empty-inline">Model coefficients will appear here when a hypothesis is selected.</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <aside className="grounding-panel">
            <div className="section-head compact">
              <div>
                <div className="eyebrow">Literature support</div>
                <h2>Ranked by resemblance</h2>
              </div>
              <div className="grounding-count">{grounding.length}</div>
            </div>
            <p className="grounding-note">
              Chunks from your uploaded corpus most similar to this hypothesis&apos;s mechanistic query (embedding
              similarity).
            </p>
            <div className="grounding-list">
              {grounding.map((item, index) => (
                <article key={item.id} className="grounding-card">
                  <div className="grounding-rank">#{index + 1}</div>
                  <div className="grounding-meta">
                    <span className={`source-badge source-${item.normalizedSource}`}>
                      {SOURCE_LABEL[item.normalizedSource] ?? item.normalizedSource}
                    </span>
                    {item.year && <span>{item.year}</span>}
                    <strong>{item.scorePct}%</strong>
                  </div>
                  <h3>{item.title}</h3>
                  <div className="reason-row">
                    {item.reasonTags.map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                </article>
              ))}
              {!grounding.length && (
                <div className="empty-inline">No corpus matches for this hypothesis yet — try a broader corpus.</div>
              )}
            </div>
          </aside>
        </section>

        <div style={{ marginTop: 24 }}>
          <button type="button" className="btn-secondary" onClick={() => navigate("/upload")}>
            Back to New Run
          </button>
        </div>
      </div>
    </div>
  );
}
