import { useMemo, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useScreeningStats } from "@/hooks/usePipeline";
import type { BundleDiagnostics, Hypothesis, ModelDiagnostics, ModelResult, ScreeningStatsBundle } from "@/types";

// ── Helpers ───────────────────────────────────────────────────────────────────

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

function fmtP(p: number): string {
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

function fmtMetric(value: unknown, digits = 4): string {
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(digits);
  if (typeof value === "string" && value) return value;
  return "—";
}

function compactMetrics(
  rows: Array<[string, unknown, number?]>,
) {
  return (
    <div className="fit-metrics">
      {rows.map(([label, value, digits]) => (
        <div className="fit-metric" key={label}>
          <small>{label}</small>
          <strong>{fmtMetric(value, digits ?? 4)}</strong>
        </div>
      ))}
    </div>
  );
}

function modelClassLabel(value?: string) {
  if (value === "time_only") return "Time-only kinetics";
  if (value === "time_plus_parameter") return "Time + parameter";
  if (value === "parameter_only") return "Parameter-only";
  return "Screening model";
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

// ── Diagnostic sub-components ─────────────────────────────────────────────────

type ScoreLevel = "poor" | "fair" | "moderate" | "good" | "strong";

function scoreLevel(s: number): ScoreLevel {
  if (s < 0.4) return "poor";
  if (s < 0.6) return "fair";
  if (s < 0.75) return "moderate";
  if (s < 0.9) return "good";
  return "strong";
}

const LEVEL_LABEL: Record<ScoreLevel, string> = {
  poor: "Poor",
  fair: "Fair",
  moderate: "Moderate",
  good: "Good",
  strong: "Strong",
};

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const level = scoreLevel(score);
  return (
    <div className="diag-score">
      <div className="diag-score-header">
        <span className="diag-score-label">Overall robustness</span>
        <span className={`diag-score-num ${level}`}>{pct}%</span>
        <span className={`diag-score-tag ${level}`}>{LEVEL_LABEL[level]}</span>
      </div>
      <div className="diag-score-track">
        <div className="diag-score-fill" data-level={level} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

type TestDef = {
  key: string;
  label: string;
  sub: string;
  fmt: (d: ModelDiagnostics) => string | null;
};

const OLS_TESTS: (TestDef & { implication: string })[] = [
  {
    key: "homoscedasticity",
    label: "Equal variance",
    sub: "Breusch-Pagan test",
    fmt: (d) => (d.breusch_pagan_p != null ? `p = ${fmtP(d.breusch_pagan_p)}` : null),
    implication:
      "Heteroscedasticity detected — OLS standard errors are biased. Significance tests unreliable; consider robust (HC3) standard errors.",
  },
  {
    key: "no_autocorrelation",
    label: "No autocorrelation",
    sub: "Durbin-Watson statistic",
    fmt: (d) => (d.durbin_watson != null ? `DW = ${d.durbin_watson.toFixed(3)}` : null),
    implication:
      "Serial correlation in residuals — standard errors are likely underestimated, inflating t-statistics. Newey-West correction advised.",
  },
  {
    key: "residual_normality",
    label: "Normal residuals",
    sub: "Jarque-Bera test",
    fmt: (d) => (d.jarque_bera_p != null ? `p = ${fmtP(d.jarque_bera_p)}` : null),
    implication:
      "Non-normal residuals — OLS inference relies on asymptotic approximation. With large n this is often acceptable; check for outliers.",
  },
  {
    key: "functional_form",
    label: "Correct specification",
    sub: "RESET test",
    fmt: (d) => (d.reset_p != null ? `p = ${fmtP(d.reset_p)}` : null),
    implication:
      "Possible functional misspecification — a nonlinear term or interaction may be missing. The linear model may underfit the true relationship.",
  },
  {
    key: "low_multicollinearity",
    label: "Low multicollinearity",
    sub: "max VIF",
    fmt: (d) => (d.max_vif != null ? `VIF = ${d.max_vif.toFixed(2)}` : null),
    implication:
      "High collinearity between predictors — coefficient estimates are unstable. Individual significance tests may be misleading; interpret jointly.",
  },
  {
    key: "no_influential_outliers",
    label: "No influential outliers",
    sub: "Cook's distance",
    fmt: (d) => (d.max_cooks_d != null ? `max D = ${d.max_cooks_d.toFixed(4)}` : null),
    implication:
      "Influential observations detected — a small subset of rows drives the results. Verify these are not data errors and check sensitivity.",
  },
];

const PANEL_TESTS: (TestDef & { implication: string })[] = [
  {
    key: "stationarity",
    label: "Series stationarity",
    sub: "ADF test (within-entity demeaned)",
    fmt: (d) =>
      d.adf_stationary_fraction != null
        ? `${Math.round(d.adf_stationary_fraction * 100)}% stationary`
        : null,
    implication:
      "Non-stationary within-entity series detected — entity FE coefficients may reflect spurious trends. Consider first-differencing or adding explicit time-trend controls.",
  },
  {
    key: "icc_adequate",
    label: "Entity effects warranted",
    sub: "Intraclass correlation ≥ 0.10",
    fmt: (d) => (d.icc != null ? `ICC = ${d.icc.toFixed(3)}` : null),
    implication:
      "Low ICC — entity fixed effects explain little between-entity variance. Pooled OLS may be sufficient; entity FE may overcorrect.",
  },
  {
    key: "fe_joint_significance",
    label: "Fixed effects significant",
    sub: "F-test for joint significance",
    fmt: (d) => (d.f_pvalue != null ? `F p = ${fmtP(d.f_pvalue)}` : null),
    implication:
      "Entity fixed effects are not jointly significant — pooled OLS may be statistically equivalent. Consider whether entity-level heterogeneity is the primary source of variation.",
  },
  {
    key: "entity_count_adequate",
    label: "Sufficient entities",
    sub: "N ≥ 10 for reliable clustered SEs",
    fmt: (d) => (d.n_entities != null ? `N = ${d.n_entities} entities` : null),
    implication:
      "Too few entities — clustered standard errors and FE estimates may be unreliable with fewer than 10 groups. Interpret significance tests with caution.",
  },
];

function AssumptionTests({ diag }: { diag: ModelDiagnostics }) {
  const isPanelFE = diag.model_family === "panel_fe";
  const tests = isPanelFE ? PANEL_TESTS : OLS_TESTS;
  return (
    <div className="diag-tests">
      {tests.map(({ key, label, sub, fmt, implication }) => {
        const value = fmt(diag);
        if (value === null) return null;
        const passed = diag.passed_tests.includes(key);
        const failed = diag.failed_tests.includes(key);
        const status = passed ? "pass" : failed ? "fail" : "na";
        return (
          <div key={key} className={`diag-test-row ${status}`}>
            <span className="diag-test-icon">{passed ? "✓" : failed ? "✗" : "—"}</span>
            <div className="diag-test-info">
              <span className="diag-test-name">{label}</span>
              <span className="diag-test-sub">{sub}</span>
              {failed && (
                <span className="diag-test-implication">{implication}</span>
              )}
            </div>
            <span className="diag-test-value">{value}</span>
          </div>
        );
      })}
    </div>
  );
}

function FamilyDiagnostics({ diagnostics }: { diagnostics: BundleDiagnostics | undefined }) {
  const families = diagnostics?.additional_families;
  if (!families || families.error) {
    return (
      <p className="diag-na-note">
        {families?.error ? String(families.error) : "Additional family diagnostics are not available for this result."}
      </p>
    );
  }
  const xgb = families.xgboost as Record<string, unknown> | undefined;
  const lasso = families.lasso as Record<string, unknown> | undefined;
  const ridge = families.ridge as Record<string, unknown> | undefined;
  const twoStage = families.two_stage as Record<string, unknown> | undefined;
  const featureImportances = (xgb?.feature_importances ?? {}) as Record<string, number | null>;
  const lassoCoefs = (lasso?.coefficients ?? {}) as Record<string, number | null>;

  return (
    <div className="family-diag-stack">
      {xgb && (
        <div className="family-diag-card">
          <div className="diag-subtitle">XGBoost</div>
          {xgb.error ? (
            <p className="diag-na-note">{String(xgb.error)}</p>
          ) : (
            <>
              {compactMetrics([
                ["in-sample R²", xgb.in_sample_r2],
                ["CV R²", xgb.cv_r2],
                ["CV std", xgb.cv_r2_std],
                ["gap", xgb.generalization_gap],
                ["RMSE", xgb.rmse],
                ["MAE", xgb.mae],
                ["features used", xgb.n_features_used, 0],
              ])}
              <div className="mini-rank-list">
                {Object.entries(featureImportances)
                  .sort((a, b) => Number(b[1] ?? 0) - Number(a[1] ?? 0))
                  .slice(0, 8)
                  .map(([name, value]) => (
                    <div className="mini-rank-row" key={name}>
                      <span title={name}>{name}</span>
                      <b>{fmtMetric(value)}</b>
                    </div>
                  ))}
              </div>
            </>
          )}
        </div>
      )}

      {lasso && (
        <div className="family-diag-card">
          <div className="diag-subtitle">LASSO</div>
          {lasso.error ? (
            <p className="diag-na-note">{String(lasso.error)}</p>
          ) : (
            <>
              {compactMetrics([
                ["alpha", lasso.selected_alpha, 6],
                ["sparsity", lasso.sparsity],
                ["nonzero", lasso.n_nonzero_coefs, 0],
                ["CV R²", lasso.cv_r2],
                ["RMSE", lasso.rmse],
                ["MAE", lasso.mae],
              ])}
              <div className="mini-rank-list">
                {Object.entries(lassoCoefs)
                  .sort((a, b) => Math.abs(Number(b[1] ?? 0)) - Math.abs(Number(a[1] ?? 0)))
                  .slice(0, 8)
                  .map(([name, value]) => (
                    <div className="mini-rank-row" key={name}>
                      <span title={name}>{name}</span>
                      <b>{fmtMetric(value)}</b>
                    </div>
                  ))}
              </div>
            </>
          )}
        </div>
      )}

      {ridge && (
        <div className="family-diag-card">
          <div className="diag-subtitle">Ridge</div>
          {ridge.error ? (
            <p className="diag-na-note">{String(ridge.error)}</p>
          ) : (
            compactMetrics([
              ["alpha", ridge.selected_alpha, 6],
              ["effective df", ridge.effective_df],
              ["shrinkage", ridge.shrinkage],
              ["condition #", ridge.condition_number],
              ["CV R²", ridge.cv_r2],
              ["gap", ridge.generalization_gap],
              ["RMSE", ridge.rmse],
              ["MAE", ridge.mae],
            ])
          )}
        </div>
      )}

      {twoStage && (
        <div className="family-diag-card">
          <div className="diag-subtitle">Two-stage</div>
          {compactMetrics([
            ["stage-1 within R²", twoStage.stage1_within_r2],
            ["stage-1 F", twoStage.stage1_f_stat],
            ["stage-1 p", twoStage.stage1_f_pvalue],
            ["stage-2 R²", twoStage.stage2_r2],
            ["stage-2 RMSE", twoStage.stage2_rmse],
            ["stage-2 MAE", twoStage.stage2_mae],
            ["stage-2 F", twoStage.stage2_f_stat],
            ["stage-2 p", twoStage.stage2_f_pvalue],
            ["entities", twoStage.n_entities, 0],
            ["stage-2 n", twoStage.stage2_n_obs, 0],
            ["obs / predictor", twoStage.stage2_obs_per_predictor],
          ])}
          {Boolean(twoStage.stage2_error) && (
            <p className="diag-na-note">{String(twoStage.stage2_error)}</p>
          )}
        </div>
      )}
    </div>
  );
}

function SelectedResult({
  hyp,
  model,
  diag,
  bundleDiagnostics,
}: {
  hyp: Hypothesis;
  model: ModelResult;
  diag: ModelDiagnostics | null;
  bundleDiagnostics?: BundleDiagnostics;
}) {
  const effects = buildEffects(model);
  const extOls = bundleDiagnostics?.extended?.ols;
  const extPanel = bundleDiagnostics?.extended?.panel;

  return (
    <div className="section-card">

      {/* ── Model identity ─────────────────────────────────────────────── */}
      <div className="sr-model-header">
        <div>
          <div className="eyebrow">selected result</div>
          <h2 style={{ marginTop: 2 }}>{hyp.hypothesis_id}: parameter effects</h2>
        </div>
        <div className="sr-model-badge">
          {bundleDiagnostics?.screening_model_class && (
            <span className="sr-model-class">
              {modelClassLabel(bundleDiagnostics.screening_model_class)}
            </span>
          )}
          <span className="sr-model-type">{model.model_type.replace(/_/g, " ").toUpperCase()}</span>
          {diag?.df_model != null && diag?.df_resid != null && (
            <span className="sr-model-df">df ({diag.df_model}, {diag.df_resid})</span>
          )}
        </div>
      </div>

      {/* ── Overall fit ─────────────────────────────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Overall fit</div>
        <div className="fit-metrics">
          <div className="fit-metric">
            <small>R²</small>
            <strong>{model.r_squared.toFixed(4)}</strong>
          </div>
          <div className="fit-metric">
            <small>{diag?.model_family === "panel_fe" ? "within R²" : "adj R²"}</small>
            <strong>{model.adj_r_squared.toFixed(4)}</strong>
          </div>
          <div className="fit-metric">
            <small>n</small>
            <strong>{model.n_observations.toLocaleString()}</strong>
          </div>
          <div className="fit-metric">
            <small>sig. vars</small>
            <strong>{model.significant_variables.length}</strong>
          </div>
          <div className="fit-metric">
            <small>F-stat</small>
            <strong>{diag?.f_statistic != null ? diag.f_statistic.toFixed(2) : "—"}</strong>
          </div>
          <div className="fit-metric">
            <small>F p-val</small>
            <strong style={{ color: (diag?.f_pvalue ?? 1) < 0.05 ? "var(--green, #22c55e)" : "var(--text)" }}>
              {diag?.f_pvalue != null ? fmtP(diag.f_pvalue) : "—"}
            </strong>
          </div>
          <div className="fit-metric">
            <small>AIC</small>
            <strong>{diag?.aic != null ? diag.aic.toFixed(1) : "—"}</strong>
          </div>
          <div className="fit-metric">
            <small>BIC</small>
            <strong>{diag?.bic != null ? diag.bic.toFixed(1) : "—"}</strong>
          </div>
        </div>
      </div>

      {/* ── Robustness score ────────────────────────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Model robustness</div>
        {diag
          ? <ScoreBar score={bundleDiagnostics?.overall_robustness_score ?? diag.diagnostic_score} />
          : <p className="diag-na-note">Run a new screening to compute robustness diagnostics.</p>
        }
      </div>

      {/* ── Statistical assumptions ─────────────────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Statistical assumptions</div>
        {diag
          ? <AssumptionTests diag={diag} />
          : <p className="diag-na-note">Diagnostic tests not yet available for this result.</p>
        }
      </div>

      {/* ── Panel / time structure ──────────────────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Panel &amp; time structure</div>
        {diag?.model_family === "panel_fe" ? (
          <div className="fit-metrics">
            {diag.n_entities != null && (
              <div className="fit-metric">
                <small>entities</small>
                <strong>{diag.n_entities}</strong>
              </div>
            )}
            <div className="fit-metric">
              <small>ICC</small>
              <strong>{diag.icc != null ? diag.icc.toFixed(3) : "—"}</strong>
            </div>
            {diag.within_r2 != null && (
              <div className="fit-metric">
                <small>within R²</small>
                <strong>{diag.within_r2.toFixed(3)}</strong>
              </div>
            )}
            {diag.between_r2 != null && (
              <div className="fit-metric">
                <small>between R²</small>
                <strong>{diag.between_r2.toFixed(3)}</strong>
              </div>
            )}
            {diag.adf_stationary_fraction != null && (
              <div className="fit-metric">
                <small>ADF stationary</small>
                <strong>{Math.round(diag.adf_stationary_fraction * 100)}%</strong>
              </div>
            )}
            {diag.f_statistic != null && (
              <div className="fit-metric">
                <small>F-stat</small>
                <strong>{diag.f_statistic.toFixed(2)}</strong>
              </div>
            )}
            {diag.f_pvalue != null && (
              <div className="fit-metric">
                <small>F p-val</small>
                <strong style={{ color: diag.f_pvalue < 0.05 ? "var(--green, #22c55e)" : "var(--text)" }}>
                  {fmtP(diag.f_pvalue)}
                </strong>
              </div>
            )}
          </div>
        ) : (
          <div className="sr-panel-note">
            <span className="sr-panel-tag">Pooled OLS</span>
            <span className="sr-panel-desc">
              No repeated-measures structure detected in this segment. Entity fixed effects and time
              trends were not estimated. If the data has multiple observations per experimental unit,
              run with panel detection enabled.
            </span>
          </div>
        )}
      </div>

      {/* ── Extended diagnostics ───────────────────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Extended OLS diagnostics</div>
        {extOls && !extOls.error ? (
          <>
            {compactMetrics([
              ["RMSE", extOls.rmse],
              ["MAE", extOls.mae],
              ["MAPE %", extOls.mape],
              ["y std", extOls.y_std],
              ["CV R²", extOls.cv_r2],
              ["CV std", extOls.cv_r2_std],
              ["obs / predictor", extOls.obs_per_predictor, 1],
              ["influential", extOls.n_influential, 0],
              ["high leverage", extOls.n_high_leverage, 0],
              ["max studentized", extOls.max_studentized],
              ["BG p", extOls.breusch_godfrey_p, 6],
              ["White p", extOls.white_p, 6],
            ])}
            {extOls.standardized_betas && (
              <div className="mini-rank-list">
                {Object.entries(extOls.standardized_betas)
                  .sort((a, b) => Math.abs(Number(b[1] ?? 0)) - Math.abs(Number(a[1] ?? 0)))
                  .slice(0, 8)
                  .map(([name, value]) => (
                    <div className="mini-rank-row" key={name}>
                      <span title={name}>std β · {name}</span>
                      <b>{fmtMetric(value)}</b>
                    </div>
                  ))}
              </div>
            )}
          </>
        ) : (
          <p className="diag-na-note">{extOls?.error ?? "Extended OLS diagnostics are unavailable."}</p>
        )}
      </div>

      <div className="diag-section">
        <div className="diag-section-title">Extended panel diagnostics</div>
        {extPanel && !extPanel.error ? (
          compactMetrics([
            ["min obs/entity", extPanel.min_obs_per_entity, 0],
            ["max obs/entity", extPanel.max_obs_per_entity, 0],
            ["mean obs/entity", extPanel.mean_obs_per_entity, 1],
            ["balanced", extPanel.balanced],
            ["Hausman p", extPanel.hausman_p, 6],
            ["FE necessary", extPanel.fe_necessary],
            ["RMSE", extPanel.rmse],
            ["MAE", extPanel.mae],
            ["AIC", extPanel.aic, 2],
            ["BIC", extPanel.bic, 2],
          ])
        ) : (
          <p className="diag-na-note">
            {extPanel?.error ?? extPanel?.fit_error ?? "Extended panel diagnostics are unavailable."}
          </p>
        )}
      </div>

      <div className="diag-section">
        <div className="diag-section-title">Additional model families</div>
        <FamilyDiagnostics diagnostics={bundleDiagnostics} />
      </div>

      {/* ── Coefficient estimates (bar chart) ──────────────────────────── */}
      <div className="diag-section">
        <div className="diag-section-title">Coefficient estimates</div>
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
              <div className={effect.coefficient >= 0 ? "effect-value positive" : "effect-value negative"}>
                {formatCoef(effect.coefficient)}
              </div>
              <div className={`effect-sig ${effect.sigLabel === "ns" ? "muted" : ""}`}>
                {effect.sigLabel}
              </div>
            </div>
          ))}
        </div>
        <p className="coef-legend">*** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns not significant</p>
      </div>

    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

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

  const bundles = useMemo(
    () => (query.data?.bundles ?? []) as ScreeningStatsBundle[],
    [query.data?.bundles],
  );
  const hypotheses = useMemo(() => bundles.map((b) => b.hypothesis), [bundles]);
  const modelResults = useMemo(() => bundles.map((b) => b.model_result), [bundles]);

  const diagMap = useMemo(() => {
    const m: Record<string, ModelDiagnostics | null> = {};
    for (const b of bundles) {
      m[b.hypothesis.id] = b.diagnostics?.ols ?? null;
    }
    return m;
  }, [bundles]);

  const selectedHyp =
    hypotheses.find((h) => h.id === selectedHypId) ??
    hypotheses.find((h) => h.hypothesis_id === selectedHypId) ??
    hypotheses[0] ??
    null;
  const selectedBundle = selectedHyp
    ? bundles.find((b) => b.hypothesis.id === selectedHyp.id) ??
      bundles.find((b) => b.hypothesis.hypothesis_id === selectedHyp.hypothesis_id)
    : undefined;

  const selectedModel = modelForHypothesis(selectedHyp, modelResults);
  const selectedDiag = selectedHyp ? (diagMap[selectedHyp.id] ?? null) : null;

  const bestR2 = useMemo(() => {
    if (!bundles.length) return 0;
    return Math.max(
      ...bundles.map((b) => {
        const diag = b.diagnostics?.ols;
        return diag?.model_family === "panel_fe"
          ? b.model_result.adj_r_squared
          : b.model_result.r_squared;
      }),
    );
  }, [bundles]);

  const hasPanelModel = useMemo(
    () => bundles.some((b) => b.diagnostics?.ols?.model_family === "panel_fe"),
    [bundles],
  );

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
              Hypotheses below are the strongest statistical associations, ranked by in-sample R².
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
              <small>
                Regime {d?.regime_id ?? regimeId} · n ={" "}
                {d?.regime_n_rows?.toLocaleString() ?? "—"}
              </small>
            </div>
            <div className="metric-card accent-metric">
              <span>Best R²</span>
              <strong>{bestR2 > 0 ? bestR2.toFixed(3) : "—"}</strong>
              <small>{hasPanelModel ? "Best within R² (panel FE)" : "Top in-sample fit in this regime"}</small>
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
          {/* Left: hypothesis list */}
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
                  const diag = diagMap[hypothesis.id];
                  const bundle = bundles.find((b) => b.hypothesis.id === hypothesis.id);
                  const selected =
                    selectedHyp?.id === hypothesis.id ||
                    (!selectedHyp && hypothesis === hypotheses[0]);
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
                        <b style={{ marginLeft: "auto" }}>
                          R² {model?.r_squared?.toFixed(3) ?? "—"}
                        </b>
                      </div>
                      <p>{hypothesis.description}</p>
                      <div className="hypothesis-variable-row">
                        <div className="chip-row">
                          {hypothesis.primary_variables.slice(0, 5).map((v) => (
                            <span key={v}>{v}</span>
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
                          <small>{diag?.model_family === "panel_fe" ? "within R²" : "adj R²"}</small>
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
                        {diag != null && (
                          <div>
                            <small>robustness</small>
                            <strong
                              className={`diag-inline-score ${scoreLevel(diag.diagnostic_score)}`}
                            >
                              {Math.round(diag.diagnostic_score * 100)}%
                            </strong>
                          </div>
                        )}
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
          </div>

          {/* Right: sticky selected result + grounding */}
          <div className="right-stack">
            {selectedHyp && selectedModel && (
              <SelectedResult
                hyp={selectedHyp}
                model={selectedModel}
                diag={selectedDiag}
                bundleDiagnostics={selectedBundle?.diagnostics}
              />
            )}

            <aside className="grounding-panel stats-grounding-ribbon">
              <div className="section-head compact">
                <div>
                  <div className="eyebrow">Next step</div>
                  <h2>Literature grounding</h2>
                </div>
              </div>
              <p className="grounding-note">
                These results show pure statistical fit. Upload PDF papers to the corpus to rank
                hypotheses by resemblance to published literature.
              </p>
              <div style={{ marginTop: 16 }}>
                <button
                  type="button"
                  className="btn-primary"
                  style={{ width: "100%" }}
                  onClick={() =>
                    groundingUrl && navigate(groundingUrl, { state: { runName } })
                  }
                >
                  Run literature grounding
                </button>
              </div>
            </aside>
          </div>
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
