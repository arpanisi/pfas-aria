import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useGroundingProgress } from "@/hooks/usePipeline";
import type {
  Citation,
  Hypothesis,
  ModelResult,
  ScreeningBundle,
} from "@/types";

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
  corpus: "Corpus",
  uploaded: "Corpus",
  arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar",
  s2: "Semantic Scholar",
  crossref: "Crossref",
  openalex: "OpenAlex",
  europe_pmc: "Europe PMC",
};

const THINK_BLOCK_RE = /<think>[\s\S]*?<\/think>/gi;
const BAD_LLM_STARTS = [
  "we need",
  "let me",
  "i need",
  "the user",
  "let's",
  "sure,",
  "here is",
  "here's",
  "rewrite",
  "input:",
  "certainly",
];
const BAD_LLM_PATTERNS = [
  "<|",
  "#include",
  ".map(",
  "Array.",
  "NSAttributed",
  "dp-thinking",
  "dp-answer",
  "---------",
  "####",
  "http://",
  "https://",
  "%20",
  "<math>",
];
const REPEATED_PUNCT_RE = /[^\w\s]{3,}/;
const COT_LABEL_RE =
  /(^|\n)\s*(think|say|body|tax|ok|end|also|but|note|for\s+force)\s*[:.]\s*/i;
const INVALID_WORD_CHARS_RE = /[*|\\@#{}_%]/;
const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "of",
  "in",
  "is",
  "are",
  "was",
  "were",
  "to",
  "for",
  "with",
  "that",
  "this",
  "it",
  "as",
  "at",
  "be",
  "by",
  "on",
  "not",
  "from",
  "but",
  "its",
  "their",
  "which",
  "have",
  "has",
  "been",
  "also",
  "both",
  "more",
  "than",
  "into",
  "over",
  "such",
  "these",
]);

function stripThinking(text: string) {
  const stripped = text.replace(THINK_BLOCK_RE, "").trim();
  for (const prefix of ["output:", "title:", "answer:"]) {
    if (stripped.toLowerCase().startsWith(prefix)) {
      return stripped.slice(prefix.length).trim();
    }
  }
  return stripped.replace(/^:+/, "").trim();
}

function isCleanLlmText(text: string, { paragraph = false } = {}) {
  const trimmed = text.trim();
  if (trimmed.length < 15) return false;
  const lower = trimmed.toLowerCase();
  if (BAD_LLM_STARTS.some((start) => lower.startsWith(start))) return false;
  if (BAD_LLM_PATTERNS.some((pattern) => trimmed.includes(pattern)))
    return false;
  if (REPEATED_PUNCT_RE.test(trimmed)) return false;
  if (COT_LABEL_RE.test(trimmed)) return false;
  const alphaSpace = Array.from(trimmed).filter((c) =>
    /[A-Za-z ]/.test(c),
  ).length;
  if (alphaSpace / trimmed.length < 0.6) return false;
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length < 6) return false;
  const dirty = words.filter((word) =>
    INVALID_WORD_CHARS_RE.test(word.replace(/[.,!?;:()[\]"']/g, "")),
  ).length;
  if (dirty / words.length > 0.15) return false;
  const counts = new Map<string, number>();
  words.forEach((word) => {
    const key = word.toLowerCase().replace(/[.,!?;:()[\]"']/g, "");
    if (key.length >= 4 && !STOPWORDS.has(key))
      counts.set(key, (counts.get(key) ?? 0) + 1);
  });
  if (counts.size && Math.max(...counts.values()) / words.length > 0.2)
    return false;
  if (paragraph && (trimmed.match(/:/g)?.length ?? 0) > 2) return false;
  return true;
}

function safeLlmText(
  text: string | null | undefined,
  fallback: string,
  opts?: { paragraph?: boolean },
) {
  const cleaned = stripThinking(text ?? "");
  return isCleanLlmText(cleaned, opts) ? cleaned : fallback;
}

function safeLlmTitle(text: string | null | undefined, fallback: string) {
  const cleaned = stripThinking(text ?? "")
    .replace(/[.]+$/, "")
    .trim();
  const words = cleaned.split(/\s+/).filter(Boolean);
  if (
    cleaned &&
    /^[A-Z]/.test(cleaned) &&
    words.length >= 5 &&
    words.length <= 12 &&
    !cleaned.includes("?") &&
    !cleaned.slice(0, -1).includes(".") &&
    isCleanLlmText(cleaned)
  ) {
    return cleaned;
  }
  return fallback;
}

function sigmoidScore(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function toTitleCase(str: string) {
  return str.replace(
    /\w\S*/g,
    (txt) => txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase(),
  );
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

function modelForHypothesis(
  hypothesis: Hypothesis | null,
  modelResults: ModelResult[],
) {
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
  const values = Object.values(model.coefficients ?? {}).map((v) =>
    Math.abs(v),
  );
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
  if (src.includes("semantic") || src === "s2") return "semantic_scholar";
  if (src.includes("arxiv")) return "arxiv";
  if (src.includes("crossref")) return "crossref";
  if (src.includes("openalex")) return "openalex";
  if (src.includes("europe") || src.includes("epmc")) return "europe_pmc";
  if (src.includes("upload") || src.includes("corpus") || src.includes("paper"))
    return "corpus";
  return src;
}

function reasonTags(citation: Citation, selected: Hypothesis | null) {
  const title = citation.title.toLowerCase();
  const tags = new Set<string>();
  if (citation.variable) tags.add(citation.variable);
  selected?.primary_variables?.slice(0, 2).forEach((v) => tags.add(v));
  if (title.includes("plasma")) tags.add("plasma reactor");
  if (title.includes("electrochemical")) tags.add("electrochemical");
  if (title.includes("degradation") || title.includes("removal"))
    tags.add("degradation");
  if (title.includes("water") || title.includes("aqueous"))
    tags.add("water treatment");
  return Array.from(tags).slice(0, 4);
}

function buildGrounding(
  citations: Citation[],
  selected: Hypothesis | null,
): GroundingCard[] {
  // Keep the best result per source (corpus, arxiv, semantic_scholar).
  const bySource = new Map<string, Citation>();
  for (const c of citations) {
    const src = normalizeSource(c.source);
    const existing = bySource.get(src);
    if (!existing || c.similarity_score > existing.similarity_score) {
      bySource.set(src, c);
    }
  }
  const ORDER = [
    "corpus",
    "arxiv",
    "semantic_scholar",
    "openalex",
    "europe_pmc",
    "crossref",
  ];
  const ordered = ORDER.flatMap((src) => {
    const c = bySource.get(src);
    return c ? [c] : [];
  });
  return ordered.map((citation) => {
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
    model.validation_passed
      ? "Validation passed"
      : "Validation requires review",
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
  const regimeId =
    regimeIdRaw != null && regimeIdRaw !== "" ? Number(regimeIdRaw) : NaN;
  const runIdParam = searchParams.get("runId")?.trim() || undefined;
  const runName =
    (location.state as { runName?: string } | null)?.runName?.trim() ||
    "Screening run";

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

  const bundles = useMemo(
    () => query.data?.bundles ?? [],
    [query.data?.bundles],
  );
  const hypotheses = useMemo(() => bundles.map((b) => b.hypothesis), [bundles]);
  const modelResults = useMemo(
    () => bundles.map((b) => b.model_result),
    [bundles],
  );

  useEffect(() => {
    if (query.error?.includes("CORPUS_TOO_SMALL")) {
      navigate("/corpus?needPapers=1", { replace: true });
    }
  }, [query.error, navigate]);

  useEffect(() => {
    if (!hypotheses.length) return;
    setSelectedHypId((prev) => {
      if (
        prev &&
        hypotheses.some((h) => h.id === prev || h.hypothesis_id === prev)
      )
        return prev;
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
      bundles.find(
        (b) => b.hypothesis.hypothesis_id === selectedHyp.hypothesis_id,
      ) ??
      null
    );
  }, [bundles, selectedHyp]);

  const selectedModel = modelForHypothesis(selectedHyp, modelResults);
  const effects = useMemo(() => buildEffects(selectedModel), [selectedModel]);
  const citationsForHyp = useMemo(
    () => selectedBundle?.citations ?? [],
    [selectedBundle?.citations],
  );
  const grounding = useMemo(
    () => buildGrounding(citationsForHyp, selectedHyp),
    [citationsForHyp, selectedHyp],
  );

  const matchScore = selectedModel?.match_score ?? 0;
  const bestR2 = modelResults.length
    ? Math.max(...modelResults.map((m) => m.r_squared))
    : 0;

  if (!filename || Number.isNaN(regimeId)) {
    return (
      <div className="results-shell">
        <div
          className="section-card"
          style={{ maxWidth: 560, margin: "48px auto" }}
        >
          <h1 className="page-title">Screening results</h1>
          <p style={{ color: "var(--text-muted)", marginBottom: 16 }}>
            Open this view from <strong>New Run</strong> after screening, or
            pass <code>?filename=…&amp;regimeId=…</code> in the URL.
          </p>
          <button
            type="button"
            className="btn-primary"
            onClick={() => navigate("/upload")}
          >
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
        <div
          className="section-card"
          style={{ maxWidth: 520, margin: "64px auto", padding: "36px 32px" }}
        >
          <h2 style={{ marginBottom: 24, fontSize: 18 }}>
            Building grounded results…
          </h2>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 8,
              alignItems: "baseline",
            }}
          >
            <span style={{ color: "var(--text-muted)", fontSize: 13 }}>
              {stage || "Starting…"}
            </span>
            <span style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
              {etaLabel && (
                <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
                  {etaLabel}
                </span>
              )}
              <span style={{ fontWeight: 600, fontSize: 13 }}>{pct}%</span>
            </span>
          </div>
          <div
            style={{
              background: "var(--border)",
              borderRadius: 8,
              height: 10,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${pct}%`,
                background: "var(--accent, #4f8ef7)",
                borderRadius: 8,
                transition: "width 0.6s ease",
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  if (query.isError && !query.error?.includes("CORPUS_TOO_SMALL")) {
    return (
      <div className="results-shell">
        <div
          className="section-card"
          style={{ maxWidth: 560, margin: "48px auto" }}
        >
          <h1 className="page-title">Could not load results</h1>
          <p style={{ color: "var(--text-muted)" }}>
            {query.error ?? "Request failed"}
          </p>
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

  if (query.isError && query.error?.includes("CORPUS_TOO_SMALL")) {
    return null;
  }

  const d = query.data;
  const safeDisplayTitle = safeLlmTitle(
    d?.display_title,
    "Literature Grounding of Discovered Signals",
  );
  const safeNextSteps = d?.next_steps
    ? safeLlmText(d.next_steps, "", { paragraph: true })
    : "";
  const safeSystemSummary = d?.system_summary
    ? safeLlmText(
        d.system_summary,
        "Screening and literature are listed for the hypotheses below.",
        { paragraph: true },
      )
    : "Screening and literature are listed for the hypotheses below.";

  return (
    <div className="results-shell">
      <div className="results-main">
        <section className="hero-card">
          <div>
            <h1>{toTitleCase(safeDisplayTitle)}</h1>
            <p>
              {safeNextSteps || (
                <>
                  Results are limited to{" "}
                  <strong>regime {d?.regime_id ?? regimeId}</strong> only.
                  Hypotheses below passed automated screening and were ranked by
                  fit and resemblance to your uploaded corpus.
                </>
              )}
            </p>
            {d?.persisted_to_run_id && (
              <p
                style={{
                  marginTop: 12,
                  fontSize: "13px",
                  color: "var(--text-muted)",
                }}
              >
                Literature-Grounded hypotheses saved for run{" "}
                <strong>#{d.persisted_to_run_id}</strong>.
              </p>
            )}
            {!runIdParam && query.data && (
              <p
                style={{
                  marginTop: 12,
                  fontSize: "13px",
                  color: "var(--text-muted)",
                }}
              >
                Pass a screening <code>runId</code> in the URL (from New Run
                after screening) to store these results in run history.
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
              <small>Ranked by Literature Relevance</small>
            </div>
            <div className="metric-card">
              <span>Best panel fit</span>
              <strong>{bestR2 > 0 ? bestR2.toFixed(3) : "—"}</strong>
              <small>Top in-sample R² in this slice</small>
            </div>
            <div className="metric-card accent-metric">
              <span>Grounding corpus</span>
              <strong>{d?.n_corpus_papers ?? "—"}</strong>
              <small>uploaded papers (minimum 3)</small>
            </div>
          </div>
        </section>

        <section className="workflow-strip">
          {["ingest", "hypothesize", "model", "validate", "ground"].map(
            (step) => (
              <div key={step} className="workflow-step active">
                <span>{step}</span>
              </div>
            ),
          )}
        </section>

        {d?.warnings?.length ? (
          <div className="section-card" style={{ marginBottom: 16 }}>
            {d.warnings.map((w) => (
              <p
                key={w}
                style={{ color: "var(--text-muted)", margin: "0 0 8px" }}
              >
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
              <div
                className="hypothesis-grid"
                style={{ gridTemplateColumns: "1fr" }}
              >
                <div
                  className="hypothesis-card selected"
                  style={{ cursor: "default" }}
                >
                  <div className="hypothesis-topline">
                    <span>Regime {d?.regime_id}</span>
                    <em>{d?.regime_n_rows?.toLocaleString() ?? "—"} rows</em>
                  </div>
                  <p>{safeSystemSummary}</p>
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
                  const bundle =
                    bundles.find((b) => b.hypothesis.id === hypothesis.id) ??
                    bundles.find(
                      (b) =>
                        b.hypothesis.hypothesis_id ===
                        hypothesis.hypothesis_id,
                    );
                  const outputVariable = bundle?.output_variable ?? null;
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
                        <b>
                          {hypothesis.is_refinement ? "refinement" : "screened"}
                        </b>
                      </div>
                      <p>
                        {safeLlmText(
                          hypothesis.description,
                          "Selected predictors are associated with the screened outcome in this regime.",
                        )}
                      </p>
                      <div className="hypothesis-variable-row">
                        <div className="chip-row">
                          {hypothesis.primary_variables
                            .slice(0, 5)
                            .map((variable) => (
                              <span key={variable}>{variable}</span>
                            ))}
                        </div>
                        {outputVariable && (
                          <span
                            className="output-variable-chip"
                            title={`Output variable: ${outputVariable}`}
                          >
                            {outputVariable}
                          </span>
                        )}
                      </div>
                      <div className="hypothesis-evidence">
                        <div>
                          <small>Model</small>
                          <strong>
                            {model?.model_type?.replace(/_/g, " ") ??
                              hypothesis.model_family}
                          </strong>
                        </div>
                        <div>
                          <small>R²</small>
                          <strong>{model?.r_squared?.toFixed(3) ?? "—"}</strong>
                        </div>
                        <div>
                          <small>Match</small>
                          <strong>
                            {sigmoidScore(
                              model?.match_score ?? hypothesis.priority_score,
                            )}
                          </strong>
                        </div>
                      </div>
                    </button>
                  );
                })}
                {!hypotheses.length && (
                  <div className="empty-inline">
                    No hypotheses met the screening and literature filters for
                    this regime.
                  </div>
                )}
              </div>
            </div>

          </div>

          <aside className="grounding-panel">
            <div className="grounding-sticky-head">
              <div className="grounding-selected-result">
                <div className="grounding-selected-title">
                  <div>
                    <div className="eyebrow">selected hypothesis</div>
                    <h2>
                      {selectedHyp?.hypothesis_id ?? "Hypothesis"}: statistical
                      evidence
                    </h2>
                  </div>
                  <div className="model-pill">
                    {selectedModel?.model_type?.replace(/_/g, " ") ??
                      "model pending"}
                  </div>
                </div>

                <div className="grounding-selected-body">
                  <div className="grounding-mini-block">
                    <h3>Hypothesis</h3>
                    <p>
                      {safeLlmText(
                        selectedHyp?.description,
                        "Selected predictors are associated with the screened outcome in this regime.",
                      )}
                    </p>
                  </div>

                  <div className="grounding-mini-block">
                    <h3>Interpretation</h3>
                    <p>
                      {safeLlmText(
                        selectedHyp?.rationale,
                        "Select a hypothesis to see how the automated fit aligns with retrieved corpus passages.",
                        { paragraph: true },
                      )}
                    </p>
                    <div className="grounding-validation-list">
                      {topValidation(selectedModel, matchScore).map((line) => (
                        <span key={line}>{line}</span>
                      ))}
                    </div>
                  </div>

                  <div className="grounding-mini-block">
                    <h3>Parameter effects</h3>
                    <div className="grounding-effect-list">
                      {effects.map((effect) => (
                        <div
                          key={effect.variable}
                          className="grounding-effect-row"
                        >
                          <div className="effect-name">{effect.variable}</div>
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
                      {!effects.length && (
                        <div className="empty-inline">
                          Model coefficients will appear here when a hypothesis
                          is selected.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="section-head compact">
                <div>
                  <div className="eyebrow">Literature support</div>
                  <h2>Ranked by resemblance</h2>
                </div>
                <div className="grounding-count">{grounding.length}</div>
              </div>
              <p className="grounding-note">
                Chunks from your uploaded corpus most similar to this
                hypothesis&apos;s mechanistic query (embedding similarity).
              </p>
            </div>
            <div className="grounding-list-label">Selected literature</div>
            <div className="grounding-list">
              {grounding.map((item, index) => (
                <article key={item.id} className="grounding-card">
                  <div className="grounding-rank">#{index + 1}</div>
                  <div className="grounding-meta">
                    <span
                      className={`source-badge source-${item.normalizedSource}`}
                    >
                      {SOURCE_LABEL[item.normalizedSource] ??
                        item.normalizedSource}
                    </span>
                    {item.year && <span>{item.year}</span>}
                    <strong>{item.scorePct}%</strong>
                  </div>
                  <h3>{item.title}</h3>
                  {item.abstract_snippet && (
                    <p className="grounding-snippet">{item.abstract_snippet}</p>
                  )}
                  <div className="reason-row">
                    {item.reasonTags.map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                </article>
              ))}
              {!grounding.length && (
                <div className="empty-inline">
                  No corpus matches for this hypothesis yet — try a broader
                  corpus.
                </div>
              )}
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
