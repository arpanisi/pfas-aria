import { useMemo, useState } from "react";
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

const DEMO_HYPOTHESES: Hypothesis[] = [
  {
    id: "demo-h1",
    hypothesis_id: "H-01",
    round: 1,
    description:
      "Surfactant-assisted plasma shifts PFCA degradation away from persistent intermediate accumulation and toward short-chain-dominated pathway states.",
    rationale:
      "The hypothesis follows from pathway metrics where entropy collapses after the intermediate-rich stage while fluoride yield increases.",
    primary_variables: ["surfactant", "time", "C2 fraction", "Shannon entropy"],
    model_family: "panel_fixed_effects",
    priority_score: 0.91,
    is_refinement: false,
  },
  {
    id: "demo-h2",
    hypothesis_id: "H-02",
    round: 1,
    description:
      "Gas atmosphere contributes an independent secondary axis of PFBS degradation performance after solution chemistry effects are accounted for.",
    rationale:
      "PCA separates gas type from bulk-solution properties; regression tests whether that axis is predictive after time effects are separated.",
    primary_variables: ["gas_used", "conductivity", "initial PFBS", "time"],
    model_family: "panel_fixed_effects",
    priority_score: 0.84,
    is_refinement: false,
  },
  {
    id: "demo-h3",
    hypothesis_id: "H-03",
    round: 2,
    description:
      "Initial chain-length composition controls maximum fluoride yield in PFCA mixtures, while surfactant addition modifies degradation kinetics.",
    rationale:
      "Mixture PCA and kinetic overlays show different drivers for defluorination and total PFCA disappearance.",
    primary_variables: ["long_chain_fraction", "short_chain_fraction", "surfactant", "k_total"],
    model_family: "panel_fixed_effects",
    priority_score: 0.79,
    is_refinement: true,
  },
];

const DEMO_MODELS: ModelResult[] = [
  {
    id: "m1",
    hypothesis_id: "H-01",
    model_type: "fixed_effect_panel",
    r_squared: 0.835,
    adj_r_squared: 0.812,
    n_observations: 186,
    coefficients: {
      time: 0.74,
      surfactant: 0.42,
      h2o2: 0.31,
      "time × surfactant": 0.18,
      "uv × h2o2": -0.12,
    },
    p_values: {
      time: 0.0001,
      surfactant: 0.008,
      h2o2: 0.022,
      "time × surfactant": 0.046,
      "uv × h2o2": 0.19,
    },
    significant_variables: ["time", "surfactant", "h2o2", "time × surfactant"],
    match_score: 0.88,
    validation_passed: true,
  },
  {
    id: "m2",
    hypothesis_id: "H-02",
    model_type: "fixed_effect_panel",
    r_squared: 0.731,
    adj_r_squared: 0.697,
    n_observations: 92,
    coefficients: {
      time: 0.61,
      gas_used: 0.39,
      conductivity: 0.28,
      additive: -0.16,
      polarity: -0.08,
    },
    p_values: {
      time: 0.0003,
      gas_used: 0.013,
      conductivity: 0.037,
      additive: 0.21,
      polarity: 0.48,
    },
    significant_variables: ["time", "gas_used", "conductivity"],
    match_score: 0.81,
    validation_passed: true,
  },
  {
    id: "m3",
    hypothesis_id: "H-03",
    model_type: "fixed_effect_panel",
    r_squared: 0.786,
    adj_r_squared: 0.761,
    n_observations: 144,
    coefficients: {
      short_chain_fraction: 0.58,
      surfactant: 0.36,
      long_chain_fraction: -0.29,
      "surfactant × composition": 0.21,
    },
    p_values: {
      short_chain_fraction: 0.004,
      surfactant: 0.018,
      long_chain_fraction: 0.041,
      "surfactant × composition": 0.066,
    },
    significant_variables: ["short_chain_fraction", "surfactant", "long_chain_fraction"],
    match_score: 0.83,
    validation_passed: true,
  },
];

const DEMO_CITATIONS: Citation[] = [
  {
    id: "c1",
    source: "corpus",
    title:
      "Development and Application of Different Non-thermal Plasma Reactors for the Removal of Perfluorosurfactants in Water",
    url: null,
    year: "2019",
    similarity_score: 0.92,
    variable: "gas_used",
  },
  {
    id: "c2",
    source: "arxiv",
    title:
      "Mechanistic Modeling of Plasma-Induced PFAS Mineralization and Intermediate Pathway Evolution",
    url: null,
    year: "2024",
    similarity_score: 0.87,
    variable: "intermediate pathway",
  },
  {
    id: "c3",
    source: "semantic_scholar",
    title:
      "Degradation of Emerging Per- and Polyfluoroalkyl Substances Using an Electrochemical Plug Flow Reactor",
    url: null,
    year: "2023",
    similarity_score: 0.84,
    variable: "time",
  },
  {
    id: "c4",
    source: "semantic_scholar",
    title:
      "Incinerability of PFOA and HFPO-DA: Mechanisms, Kinetics, and Thermal Stability Ranking",
    url: null,
    year: "2023",
    similarity_score: 0.78,
    variable: "radical pathway",
  },
];

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

export function Dashboard() {
  const { runId } = useParams<{ runId: string }>();
  const [selectedHypId, setSelectedHypId] = useState<string | null>(null);
  const [activeRound, setActiveRound] = useState<number | undefined>(undefined);

  const { lastMessage } = useWebSocket(runId ?? null);
  const { data: summary } = useRunSummary(runId ?? null);
  const { data: status } = useRunStatus(runId ?? null, true);
  const { data: apiHypotheses = [] } = useHypotheses(runId ?? null, activeRound);
  const { data: apiModelResults = [] } = useModelResults(runId ?? null);
  const { data: apiCitations = [] } = useCitations(runId ?? null);
  const { data: convergence = [] } = useConvergence(runId ?? null);

  const hypotheses = apiHypotheses.length ? apiHypotheses : DEMO_HYPOTHESES;
  const modelResults = apiModelResults.length ? apiModelResults : DEMO_MODELS;
  const citations = apiCitations.length ? apiCitations : DEMO_CITATIONS;

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
  const runStatus = status?.status ?? (apiHypotheses.length ? "completed" : "demo");
  const maxRound = summary?.n_rounds ?? Math.max(...hypotheses.map((h) => h.round), 0);
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
          <div className={`status-badge badge-${runStatus}`}>{statusStage(runStatus, lastMessage?.stage)}</div>
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
              <div className="hypothesis-grid">
                {hypotheses.map((hypothesis) => {
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
                  <p>{selectedHyp?.rationale ?? "The agent is separating time evolution from treatment-factor effects, then testing whether the observed coefficient pattern is consistent with the proposed mechanism."}</p>
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
