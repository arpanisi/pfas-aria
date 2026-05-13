import { useEffect, useMemo, useState } from "react";
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
import type { Citation, Hypothesis, ModelResult } from "@/types";

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

/** Max hypotheses to show when a run has more than this many (top by score, else stable sample). */
const DISPLAY_TOP_N = 3;

function strHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = Math.imul(31, h) + s.charCodeAt(i) | 0;
  return h >>> 0;
}

/** Stable pseudo-random subset when ranking is unavailable. */
function pickSeededHypothesisSample(runId: string, hyps: Hypothesis[], n: number): Hypothesis[] {
  if (hyps.length <= n) return hyps;
  return [...hyps]
    .sort((a, b) => strHash(`${runId}:${a.id}`) - strHash(`${runId}:${b.id}`))
    .slice(0, n);
}

function resolveModelForHypothesis(h: Hypothesis, models: ModelResult[]): ModelResult | null {
  return (
    models.find((m) => m.hypothesis_id === h.id) ??
    models.find((m) => m.hypothesis_id === h.hypothesis_id) ??
    null
  );
}

function jointRankingScore(m: ModelResult): number {
  const r2 = Math.max(0, m.adj_r_squared ?? m.r_squared ?? 0);
  const lit = Math.max(0, m.match_score ?? 0);
  return r2 * (0.5 + lit);
}

/**
 * When there are more than DISPLAY_TOP_N hypotheses, show the best N by
 * (adjusted) R² × literature match. If no model rows join, use a stable sample.
 */
function pickDisplayHypotheses(
  runId: string,
  hyps: Hypothesis[],
  models: ModelResult[],
  n: number,
): { display: Hypothesis[]; usedRanking: boolean; total: number } {
  const total = hyps.length;
  if (hyps.length <= n) {
    return { display: hyps, usedRanking: false, total };
  }
  const scored = hyps.map((h) => {
    const m = resolveModelForHypothesis(h, models);
    return { h, s: m ? jointRankingScore(m) : -1 };
  });
  const hasRankable = scored.some((x) => x.s >= 0);
  if (hasRankable) {
    scored.sort((a, b) => {
      if (b.s !== a.s) return b.s - a.s;
      return a.h.id.localeCompare(b.h.id);
    });
    return {
      display: scored.slice(0, n).map((x) => x.h),
      usedRanking: true,
      total,
    };
  }
  return {
    display: pickSeededHypothesisSample(runId, hyps, n),
    usedRanking: false,
    total,
  };
}

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
  if (!hypothesis) return null;
  return resolveModelForHypothesis(hypothesis, modelResults);
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
  const title = (citation.title ?? "").toLowerCase();
  const tags = new Set<string>();
  if (citation.variable) tags.add(citation.variable);
  selected?.primary_variables?.slice(0, 2).forEach((v) => tags.add(v));
  if (title.includes("plasma")) tags.add("plasma reactor");
  if (title.includes("electrochemical")) tags.add("kinetics");
  if (title.includes("incinerability") || title.includes("thermal")) tags.add("mechanism");
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
  if (!model) return ["Waiting for model evidence", "Grounding not yet available"];
  const lines = [
    `${model.model_type.replace(/_/g, " ")} · n=${model.n_observations}`,
    `R² ${model.r_squared.toFixed(3)} · adjusted R² ${model.adj_r_squared.toFixed(3)}`,
    model.validation_passed ? "Validation passed" : "Validation requires review",
    `Literature resemblance ${sigmoidScore(matchScore || model.match_score)}`,
  ];
  return lines;
}

function statusStage(status?: string, wsStage?: string) {
  if (wsStage) return wsStage.replace(/_/g, " ");
  return status?.replace(/_/g, " ") ?? "loading";
}

/** Map API status strings to existing badge-* CSS classes. */
function statusBadgeClass(status: string) {
  if (status === "loading") return "badge-loading";
  if (status === "converged") return "badge-converged";
  if (status === "failed") return "badge-failed";
  if (status.includes("running")) return "badge-running";
  return "badge-completed";
}

export function Dashboard() {
  const { runId } = useParams<{ runId: string }>();
  const [selectedHypId, setSelectedHypId] = useState<string | null>(null);
  const [activeRound, setActiveRound] = useState<number | undefined>(undefined);

  const { lastMessage } = useWebSocket(runId ?? null);
  const { data: summary } = useRunSummary(runId ?? null);
  const { data: status } = useRunStatus(runId ?? null, true);
  const { data: apiHypotheses = [], isFetched: hypothesesFetched } = useHypotheses(
    runId ?? null,
    activeRound,
  );
  const { data: apiModelResults = [], isFetched: modelsFetched } = useModelResults(runId ?? null);
  const { data: apiCitations = [] } = useCitations(runId ?? null);
  const { data: convergence = [] } = useConvergence(runId ?? null);

  const { displayHypotheses, subsetNote } = useMemo(() => {
    if (!runId || !hypothesesFetched) {
      return { displayHypotheses: [] as Hypothesis[], subsetNote: null as string | null };
    }
    const all = apiHypotheses;
    if (all.length === 0) {
      return { displayHypotheses: [], subsetNote: null };
    }
    if (!modelsFetched) {
      return { displayHypotheses: all, subsetNote: null };
    }
    const { display, usedRanking, total } = pickDisplayHypotheses(
      runId,
      all,
      apiModelResults,
      DISPLAY_TOP_N,
    );
    if (display.length >= total) {
      return { displayHypotheses: display, subsetNote: null };
    }
    const note = usedRanking
      ? `Showing top ${DISPLAY_TOP_N} of ${total} tested hypotheses (ranked by adjusted R² × literature match).`
      : `Showing ${DISPLAY_TOP_N} of ${total} tested hypotheses (stable sample — link models to rank).`;
    return { displayHypotheses: display, subsetNote: note };
  }, [
    runId,
    hypothesesFetched,
    apiHypotheses,
    modelsFetched,
    apiModelResults,
  ]);

  const hypotheses = displayHypotheses;
  const modelResults = apiModelResults;
  const citations = apiCitations;

  const showHypothesesLoading = Boolean(runId && !hypothesesFetched);
  const showHypothesesEmpty =
    Boolean(runId && hypothesesFetched && apiHypotheses.length === 0);

  useEffect(() => {
    const ids = new Set(hypotheses.map((h) => h.id));
    if (selectedHypId && !ids.has(selectedHypId)) {
      setSelectedHypId(hypotheses[0]?.id ?? null);
    }
  }, [hypotheses, selectedHypId]);

  const selectedHyp =
    hypotheses.find((h) => h.id === selectedHypId) ??
    hypotheses.find((h) => h.hypothesis_id === selectedHypId) ??
    hypotheses[0] ??
    null;

  const selectedModel = modelForHypothesis(selectedHyp, modelResults);
  const effects = useMemo(() => buildEffects(selectedModel), [selectedModel]);
  const grounding = useMemo(() => buildGrounding(citations, selectedHyp), [citations, selectedHyp]);

  const matchScore = lastMessage?.match_score ?? status?.final_match_score ?? selectedModel?.match_score ?? 0;
  const currentRound = lastMessage?.round ?? status?.current_round ?? selectedHyp?.round ?? 0;
  const runStatus =
    status?.status ??
    summary?.status ??
    (hypothesesFetched ? (apiHypotheses.length ? "completed" : "no_hypotheses") : "loading");
  const maxRound =
    summary?.n_rounds ??
    Math.max(0, ...apiHypotheses.map((h) => h.round));
  const rounds = Array.from({ length: maxRound }, (_, i) => i + 1);
  const validation = topValidation(selectedModel, matchScore);
  const bestConvergence = convergence.length ? convergence[convergence.length - 1] : undefined;

  return (
    <div className="results-shell">
      <div className="results-main">
        <section className="hero-card">
          <div>
            <div className="eyebrow">Agentic hypothesis testing dashboard</div>
            <h1>{summary?.run_name ?? "PFAS mechanistic inference run"}</h1>
            <p>
              Hypotheses are tested statistically, interpreted mechanistically, and grounded against a mixed literature corpus of uploaded papers plus externally retrieved matches.
            </p>
          </div>
          <div className="hero-metrics">
            <div className="metric-card">
              <span>Round</span>
              <strong>{currentRound || "—"}</strong>
              <small>of {summary?.n_rounds ?? (maxRound || "?")}</small>
            </div>
            <div className="metric-card">
              <span>Outcome</span>
              <strong>{summary?.outcome_variable ?? "dynamic"}</strong>
              <small>{summary?.selected_features?.length ?? "—"} selected features</small>
            </div>
            <div className="metric-card accent-metric">
              <span>Resemblance</span>
              <strong>{sigmoidScore(matchScore)}</strong>
              <small>{bestConvergence ? `best R² ${bestConvergence.best_r_squared.toFixed(2)}` : "literature match"}</small>
            </div>
          </div>
        </section>

        <section className="workflow-strip">
          {["ingest", "hypothesize", "model", "validate", "ground"].map((step) => (
            <div key={step} className={`workflow-step ${statusStage(runStatus, lastMessage?.stage).includes(step) ? "active" : ""}`}>
              <span>{step}</span>
            </div>
          ))}
          <div className={`status-badge ${statusBadgeClass(runStatus)}`}>{statusStage(runStatus, lastMessage?.stage)}</div>
        </section>

        <section className="content-grid">
          <div className="left-stack">
            <div className="section-card">
              <div className="section-head">
                <div>
                  <div className="eyebrow">tested claims</div>
                  <h2>Hypotheses</h2>
                </div>
                {maxRound > 0 && (
                  <div className="round-filter">
                    <button className={activeRound === undefined ? "active" : ""} onClick={() => setActiveRound(undefined)}>All</button>
                    {rounds.map((r) => (
                      <button key={r} className={activeRound === r ? "active" : ""} onClick={() => setActiveRound(r)}>R{r}</button>
                    ))}
                  </div>
                )}
              </div>
              {subsetNote && (
                <p className="subset-sample-note" role="status">
                  {subsetNote}
                </p>
              )}
              <div className="hypothesis-grid">
                {showHypothesesLoading && (
                  <div className="empty-inline dashboard-loading">Loading hypotheses for this run…</div>
                )}
                {showHypothesesEmpty && (
                  <div className="empty-inline">
                    No hypotheses are stored for this run yet. Complete screening with persistence, or open a run from history that has saved results.
                  </div>
                )}
                {!showHypothesesLoading &&
                  !showHypothesesEmpty &&
                  hypotheses.map((hypothesis) => {
                  const model = modelForHypothesis(hypothesis, modelResults);
                  const selected = selectedHyp?.id === hypothesis.id;
                  return (
                    <button
                      key={hypothesis.id}
                      className={`hypothesis-card ${selected ? "selected" : ""}`}
                      onClick={() => setSelectedHypId(hypothesis.id)}
                    >
                      <div className="hypothesis-topline">
                        <span>{hypothesis.hypothesis_id}</span>
                        <em>Round {hypothesis.round}</em>
                        <b>{hypothesis.is_refinement ? "refinement" : "tested"}</b>
                      </div>
                      <p>{hypothesis.description}</p>
                      <div className="chip-row">
                        {hypothesis.primary_variables.slice(0, 5).map((variable) => (
                          <span key={variable}>{variable}</span>
                        ))}
                      </div>
                      <div className="hypothesis-evidence">
                        <div><small>Model</small><strong>{model?.model_type?.replace(/_/g, " ") ?? hypothesis.model_family}</strong></div>
                        <div><small>R²</small><strong>{model?.r_squared?.toFixed(3) ?? "—"}</strong></div>
                        <div><small>Match</small><strong>{sigmoidScore(model?.match_score ?? hypothesis.priority_score)}</strong></div>
                      </div>
                    </button>
                  );
                })}
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
                      (hypothesesFetched
                        ? "No interpretation text is stored for this hypothesis yet. Run literature-grounded screening with persistence, or re-run the modeling pipeline so the narrative layer can populate rationale."
                        : "Loading…")}
                  </p>
                  <div className="validation-list">
                    {validation.map((line) => <span key={line}>{line}</span>)}
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
                    {!effects.length && <div className="empty-inline">Model coefficients will appear here after the modeling agent finishes.</div>}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <aside className="grounding-panel">
            <div className="section-head compact">
              <div>
                <div className="eyebrow">retrieval + ranking</div>
                <h2>Top explanatory literature</h2>
              </div>
              <div className="grounding-count">{grounding.length}</div>
            </div>
            <p className="grounding-note">
              Ranked by mechanistic resemblance across uploaded corpus and auto-expanded scholarly retrieval.
            </p>
            <div className="grounding-list">
              {grounding.map((item, index) => (
                <article key={item.id} className="grounding-card">
                  <div className="grounding-rank">#{index + 1}</div>
                  <div className="grounding-meta">
                    <span className={`source-badge source-${item.normalizedSource}`}>{SOURCE_LABEL[item.normalizedSource] ?? item.normalizedSource}</span>
                    {item.year && <span>{item.year}</span>}
                    <strong>{item.scorePct}%</strong>
                  </div>
                  <h3>{item.title}</h3>
                  <div className="reason-row">
                    {item.reasonTags.map((tag) => <span key={tag}>{tag}</span>)}
                  </div>
                </article>
              ))}
              {!grounding.length && <div className="empty-inline">The grounding agent has not returned literature matches yet.</div>}
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}
