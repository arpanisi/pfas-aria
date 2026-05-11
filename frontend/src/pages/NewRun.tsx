import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { getCorpusStats, getLegacySegmentationPreview } from "@/api/endpoints";
import { useAutomatedScreeningIteration, useUploadDataset } from "@/hooks/usePipeline";
import {
  clearNewRunDraft,
  loadNewRunDraft,
  saveNewRunDraft,
} from "@/lib/newRunDraft";
import type { ColumnInfo, DatasetPreview, LegacyRegimeSummary } from "@/types";

type Step = "upload" | "configure" | "hypotheses";

type UploadLocationState = { preview?: DatasetPreview };

export function NewRun() {
  const navigate = useNavigate();
  const location = useLocation();
  const { mutateAsync: upload, isPending: uploading } = useUploadDataset();
  const screeningMutation = useAutomatedScreeningIteration();

  const [step, setStep] = useState<Step>("upload");
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [outcome, setOutcome] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [runName, setRunName] = useState("");
  const [threshold, setThreshold] = useState(0.75);
  const [hypothesesTestedCount, setHypothesesTestedCount] = useState<number | null>(null);
  const [screeningRunId, setScreeningRunId] = useState<string | null>(null);

  const [progressMsgIdx, setProgressMsgIdx] = useState(0);
  const progressMessages = [
    "Scanning inputs, outputs, and regime structure…",
    "Exploring variable mixes and multicollinearity patterns…",
    "Building the hypothesis space across regimes…",
    "Fitting statistical models and counting tests…",
  ] as const;

  useEffect(() => {
    if (!screeningMutation.isPending) return;
    setProgressMsgIdx(0);
    const id = window.setInterval(() => {
      setProgressMsgIdx((i) => (i + 1) % progressMessages.length);
    }, 2200);
    return () => window.clearInterval(id);
  }, [screeningMutation.isPending, progressMessages.length]);

  const hydrateOnceRef = useRef(false);

  const applyPreviewFromUpload = useCallback((p: DatasetPreview) => {
    setPreview(p);
    const nums = p.columns.filter((c: ColumnInfo) => c.is_numeric).map((c: ColumnInfo) => c.name);
    setFeatures(nums);
    setStep("configure");
  }, []);

  useEffect(() => {
    const navPreview = (location.state as UploadLocationState | null)?.preview;
    if (navPreview) {
      applyPreviewFromUpload(navPreview);
      navigate(location.pathname, { replace: true, state: {} });
      hydrateOnceRef.current = true;
      return;
    }
    if (hydrateOnceRef.current) return;
    hydrateOnceRef.current = true;
    const draft = loadNewRunDraft();
    if (!draft) return;
    setPreview(draft.preview);
    setStep(draft.step);
    setOutcome(draft.outcome);
    setFeatures(draft.features);
    setRunName(draft.runName);
    setThreshold(draft.threshold);
    setScreeningRunId(draft.screeningRunId ?? null);
  }, [location.state, location.pathname, navigate, applyPreviewFromUpload]);

  useEffect(() => {
    if (!preview) return;
    saveNewRunDraft({
      v: 1,
      preview,
      step,
      outcome,
      features,
      runName,
      threshold,
      screeningRunId,
    });
  }, [preview, step, outcome, features, runName, threshold, screeningRunId]);

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    try {
      const p = await upload(files[0]);
      applyPreviewFromUpload(p);
      toast.success(`Loaded ${p.n_rows.toLocaleString()} rows`);
    } catch { toast.error("Upload failed"); }
  }, [upload, applyPreviewFromUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"], "text/tab-separated-values": [".tsv"] },
    maxFiles: 1,
    disabled: uploading,
  });

  const [selectedRegimeId, setSelectedRegimeId] = useState<number | null>(null);

  const segQuery = useQuery({
    queryKey: ["legacySegmentation", preview?.filename],
    queryFn: () => getLegacySegmentationPreview(preview!.filename),
    enabled:
      !!preview?.filename && (step === "configure" || step === "hypotheses"),
    staleTime: 60_000,
    retry: 0,
  });

  useEffect(() => {
    const regimes = segQuery.data?.regimes;
    if (!regimes?.length) return;
    setSelectedRegimeId((prev) => {
      const ids = new Set(regimes.map((r) => r.regime_id));
      if (prev != null && ids.has(prev)) return prev;
      return regimes[0].regime_id;
    });
  }, [segQuery.data]);

  useEffect(() => {
    if (step !== "configure" || !preview) return;
    if (!segQuery.isFetched) return;
    const d = segQuery.data;
    const nums = preview.columns.filter((c: ColumnInfo) => c.is_numeric).map((c: ColumnInfo) => c.name);
    if (!d?.regimes?.length || (!(d.input_cols?.length) && !(d.output_cols?.length))) {
      const oc = outcome && nums.includes(outcome) ? outcome : nums[0] ?? "";
      setOutcome(oc);
      setFeatures(nums.filter((n) => n !== oc));
      return;
    }
    const rid = selectedRegimeId ?? d.regimes[0].regime_id;
    const regime = d.regimes.find((r) => r.regime_id === rid);
    if (!regime) return;
    const vi = new Set(regime.non_constant_input_cols);
    const varyingIn = d.input_cols.filter((n) => vi.has(n));
    const vo = new Set(regime.non_constant_output_cols);
    const varyingOut = d.output_cols.filter((n) => vo.has(n));
    const oc = varyingOut[0] ?? d.output_cols[0] ?? nums[0] ?? "";
    setOutcome(oc);
    const ins = varyingIn.length ? varyingIn : d.input_cols;
    setFeatures(ins.filter((n) => n !== oc));
  }, [step, preview, segQuery.isFetched, segQuery.data, selectedRegimeId, outcome]);

  const columnByName = useMemo(() => {
    const m = new Map<string, ColumnInfo>();
    if (preview) for (const c of preview.columns) m.set(c.name, c);
    return m;
  }, [preview]);

  const selectedRegime = useMemo((): LegacyRegimeSummary | null => {
    if (!segQuery.data || selectedRegimeId == null) return null;
    return segQuery.data.regimes.find((r) => r.regime_id === selectedRegimeId) ?? null;
  }, [segQuery.data, selectedRegimeId]);

  const regimeVaryingInputColumns = useMemo(() => {
    if (!selectedRegime || !segQuery.data?.input_cols?.length || !preview) return [];
    const allowed = new Set(selectedRegime.non_constant_input_cols);
    return segQuery.data.input_cols
      .filter((name) => allowed.has(name))
      .map((name) => columnByName.get(name))
      .filter((c): c is ColumnInfo => c != null);
  }, [segQuery.data, preview, columnByName, selectedRegime]);

  const regimeVaryingOutputColumns = useMemo(() => {
    if (!selectedRegime || !segQuery.data?.output_cols?.length || !preview) return [];
    const allowed = new Set(selectedRegime.non_constant_output_cols);
    return segQuery.data.output_cols
      .filter((name) => allowed.has(name))
      .map((name) => columnByName.get(name))
      .filter((c): c is ColumnInfo => c != null);
  }, [segQuery.data, preview, columnByName, selectedRegime]);

  const showRegimeColumnPanel = Boolean(
    segQuery.data?.regimes?.length &&
      selectedRegime &&
      ((segQuery.data.input_cols?.length ?? 0) > 0 || (segQuery.data.output_cols?.length ?? 0) > 0),
  );

  const screeningCombinationsPreview = useMemo(() => {
    const d = segQuery.data;
    if (!d?.regimes?.length || !preview) return [];
    const lines: string[] = [];
    const hasPanel =
      preview.columns.some((c) => /experiment|batch|^run$/i.test(c.name)) &&
      preview.columns.some((c) => /time|day|hour/i.test(c.name));
    const mgInputs = d.input_cols.filter((c) => {
      const lo = c.toLowerCase();
      return lo.includes("mg/l") || lo.includes("mg_l");
    });
    const numericInputs = d.input_cols.filter((n) => columnByName.get(n)?.is_numeric);
    const trunc = (xs: string[], max = 5) =>
      xs.length <= max
        ? xs.join(", ")
        : `${xs.slice(0, max).join(", ")} … (+${xs.length - max} more)`;
    const numericOutputs = d.output_cols.filter((n) => columnByName.get(n)?.is_numeric);
    const regimesForPreview =
      selectedRegimeId != null
        ? d.regimes.filter((r) => r.regime_id === selectedRegimeId)
        : d.regimes;
    for (const r of regimesForPreview) {
      const outsForRegime = numericOutputs.filter((o) => {
        if (!r.non_constant_output_cols.length) return true;
        return r.non_constant_output_cols.includes(o);
      });
      const insForRegime = mgInputs.filter((c) => {
        if (!r.non_constant_input_cols.length) return true;
        return r.non_constant_input_cols.includes(c);
      });
      let inputSide: string[];
      let usedNumericPool = false;
      if (insForRegime.length >= 2) inputSide = insForRegime;
      else if (mgInputs.length >= 2) inputSide = mgInputs;
      else if (numericInputs.length >= 2) {
        inputSide = numericInputs;
        usedNumericPool = true;
      } else continue;
      const panelNote = hasPanel
        ? " · panel-style fits when experiment + time columns exist"
        : "";
      const inputLabel = usedNumericPool ? "numeric predictors" : "mg/L predictors";
      for (const o of outsForRegime) {
        lines.push(
          `Regime ${r.regime_id} → ${o} · ${inputLabel}: ${trunc(inputSide)}${panelNote}`,
        );
      }
    }
    return lines;
  }, [segQuery.data, preview, columnByName, selectedRegimeId]);

  const segmentationReady = Boolean(
    preview &&
      segQuery.isFetched &&
      !segQuery.isError &&
      (segQuery.data?.regimes?.length ?? 0) > 0 &&
      selectedRegimeId != null,
  );

  const configureCanContinue = segmentationReady;

  const renderReadonlyColCard = (col: ColumnInfo) => (
    <div key={col.name} className="col-card col-card-readonly">
      <div className="col-card-header">
        <span className="col-name">{col.name}</span>
        <span className={`col-dtype ${col.is_numeric ? "num" : ""}`}>
          {col.is_numeric ? "num" : "cat"}
        </span>
      </div>
      <div className="col-meta">
        {col.n_unique} unique · {col.missing_pct.toFixed(1)}% missing
      </div>
      <div className="col-sample">{col.sample_values.slice(0, 3).map(String).join(", ")}</div>
    </div>
  );

  const steps: Step[] = ["upload", "configure", "hypotheses"];

  const handleLaunch = async () => {
    if (!preview || selectedRegimeId == null) {
      toast.error("Select a regime on the Configure step.");
      return;
    }
    setHypothesesTestedCount(null);
    try {
      const r = await screeningMutation.mutateAsync({
        filename: preview.filename,
        run_name: runName.trim() || "Untitled screening",
        regime_id: selectedRegimeId,
        convergence_threshold: threshold,
      });
      setHypothesesTestedCount(r.hypotheses_tested);
      setScreeningRunId(r.run_id ?? null);
      toast.success("Screening iteration complete");
    } catch {
      toast.error("Screening failed");
    }
  };

  const handleViewScreeningResults = async () => {
    if (!preview || selectedRegimeId == null) {
      toast.error("Missing dataset or regime.");
      return;
    }
    try {
      const stats = await getCorpusStats();
      if ((stats.n_papers ?? 0) < 3) {
        toast.error("Upload at least three PDF papers to the corpus to run literature grounding.");
        navigate("/corpus?needPapers=1");
        return;
      }
      const q = new URLSearchParams({
        filename: preview.filename,
        regimeId: String(selectedRegimeId),
      });
      if (screeningRunId) q.set("runId", screeningRunId);
      navigate(`/runs/screening?${q.toString()}`, {
        state: { runName: runName.trim() || "Untitled screening" },
      });
    } catch {
      toast.error("Could not verify corpus. Try again.");
    }
  };

  return (
    <div className="new-run-page">
      <div className="page-header">
        <h1 className="page-title">New Run</h1>
        <div className="page-header-right">
          {preview && (step === "configure" || step === "hypotheses") && (
            <button
              type="button"
              className="btn-secondary new-run-header-refresh"
              onClick={() => void segQuery.refetch()}
              disabled={segQuery.isFetching}
            >
              {segQuery.isFetching ? "Loading…" : "Refresh"}
            </button>
          )}
          <div className="step-pills">
            {(["Upload", "Configure", "Hypotheses"] as const).map((s, i) => {
              const cur = steps.indexOf(step);
              return (
                <div key={s} className={`step-pill ${i === cur ? "active" : i < cur ? "done" : ""}`}>{s}</div>
              );
            })}
          </div>
        </div>
      </div>

      {step === "upload" && (
        <div className="new-run-upload">
          {preview && (
            <div className="dataset-bar" style={{ marginBottom: 16 }}>
              <span>{preview.filename}</span>
              <span>{preview.n_rows.toLocaleString()} rows</span>
              <span>{preview.n_cols} columns</span>
              <button
                type="button"
                className="btn-secondary"
                style={{ marginLeft: "auto" }}
                onClick={() => {
                  clearNewRunDraft();
                  setPreview(null);
                  setOutcome("");
                  setFeatures([]);
                  setHypothesesTestedCount(null);
                  setScreeningRunId(null);
                }}
              >
                Replace dataset
              </button>
            </div>
          )}
          <div {...getRootProps()} className={`dropzone-large ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="dz-icon">⬆</div>
            <div className="dz-text">{uploading ? "Uploading..." : "Drop your dataset here"}</div>
            <div className="dz-hint">CSV, TSV, Excel · up to 100k rows</div>
          </div>
        </div>
      )}

      {step === "configure" && preview && (
        <div className="new-run-configure">
          <div className="dataset-bar">
            <span>{preview.filename}</span>
            <span>{preview.n_rows.toLocaleString()} rows</span>
            <span>{preview.n_cols} columns</span>
          </div>

          <div className="new-run-regime-block">
            {segQuery.isLoading && <div className="seg-preview-status">Loading…</div>}
            {segQuery.isError && (
              <div className="seg-preview-error">
                {(() => {
                  const d = (segQuery.error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
                  if (typeof d === "string") return d;
                  if (Array.isArray(d)) return d.map((x) => (typeof x === "object" && x && "msg" in x ? String((x as { msg: string }).msg) : JSON.stringify(x))).join("; ");
                  return "Could not load preview for this file.";
                })()}
              </div>
            )}
            {segQuery.data && (
              <>
                {segQuery.data.n_regimes > 0 && (
                  <p className="seg-regime-intro">
                    PFAS-ARIA intelligently discovers{" "}
                    <span className="seg-regime-intro-n">{segQuery.data.n_regimes}</span>{" "}
                    {segQuery.data.n_regimes === 1 ? "regime" : "regimes"} within the data
                    based on common behavior
                  </p>
                )}
                <div className="seg-regime-tabs" role="tablist" aria-label="Regimes">
                  {segQuery.data.regimes.map((r) => {
                    const active = r.regime_id === selectedRegimeId;
                    return (
                      <button
                        key={r.regime_id}
                        type="button"
                        role="tab"
                        aria-selected={active}
                        aria-label={`Regime ${r.regime_id}, ${r.n_rows.toLocaleString()} rows`}
                        className={`seg-regime-tab${active ? " active" : ""}`}
                        onClick={() => setSelectedRegimeId(r.regime_id)}
                      >
                        <span
                          className="seg-regime-tab-cond"
                          title={
                            r.condition_values_sample.length > 0
                              ? r.condition_values_sample.join(" · ")
                              : undefined
                          }
                        >
                          {r.condition_values_sample.length > 0
                            ? r.condition_values_sample.join(" · ")
                            : "—"}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          <div className="col-grid-scroll">
            <div className="regime-column-panels">
              {showRegimeColumnPanel && (
                <>
                  <section className="regime-col-section" aria-labelledby="regime-inputs-heading">
                    <h3 id="regime-inputs-heading" className="regime-col-section-title">
                      Inputs
                      {selectedRegime && (
                        <span className="regime-col-section-meta">
                          {" "}
                          · regime {selectedRegime.regime_id} · {selectedRegime.n_rows.toLocaleString()} rows
                          {" "}
                          · varying only
                        </span>
                      )}
                    </h3>
                    {regimeVaryingInputColumns.length > 0 ? (
                      <div className="col-grid">{regimeVaryingInputColumns.map(renderReadonlyColCard)}</div>
                    ) : (
                      <p className="regime-col-section-empty">No varying inputs in this regime.</p>
                    )}
                  </section>
                  <section className="regime-col-section" aria-labelledby="regime-outputs-heading">
                    <h3 id="regime-outputs-heading" className="regime-col-section-title">
                      Outputs
                    </h3>
                    {regimeVaryingOutputColumns.length > 0 ? (
                      <div className="col-grid">{regimeVaryingOutputColumns.map(renderReadonlyColCard)}</div>
                    ) : (
                      <p className="regime-col-section-empty">No varying outputs in this regime.</p>
                    )}
                  </section>
                </>
              )}
            </div>
          </div>
          <div className="step-actions" style={{ marginTop: 16, flexShrink: 0 }}>
            <button className="btn-secondary" onClick={() => setStep("upload")}>Back</button>
            <button
              className="btn-primary"
              onClick={() => {
                setHypothesesTestedCount(null);
                setStep("hypotheses");
              }}
              disabled={!configureCanContinue}
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      {step === "hypotheses" && preview && (
        <div className="new-run-hypotheses">
          <div className="dataset-bar" style={{ marginBottom: 16 }}>
            <span>{preview.filename}</span>
            <span>{preview.n_rows.toLocaleString()} rows</span>
            <span>{preview.n_cols} columns</span>
          </div>
          <p className="hyp-run-lead">
            <strong>PFAS-ARIA</strong> screens inputs, output targets, and model families for{" "}
            <strong>
              {selectedRegimeId != null ? `regime ${selectedRegimeId}` : "your selected regime"}
            </strong>{" "}
            only — the regime you chose on Configure. The list below is limited to that slice.
          </p>
          <div className="settings-grid">
            <div className="field field-span-2">
              <label>Run name</label>
              <input
                className="input"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="e.g. PFAS UV Batch 1"
              />
            </div>
            <div className="field field-span-2">
              <label>Convergence threshold</label>
              <input
                className="input"
                type="number"
                min={0.5}
                max={1.0}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
              <span className="field-hint">Used for downstream ranking when the full agent pipeline runs.</span>
            </div>
          </div>

          {screeningMutation.isPending && (
            <div className="hyp-run-progress" role="status" aria-live="polite">
              <div className="hyp-run-progress-bar indeterminate" />
              <p className="hyp-run-progress-text">{progressMessages[progressMsgIdx]}</p>
            </div>
          )}

          <div className="run-summary-box">
            <p className="run-screening-plan-title">
              Planned input → output combinations (regime {selectedRegimeId ?? "—"})
            </p>
            {!segmentationReady ? (
              <p className="run-screening-dataset">Loading regime layout…</p>
            ) : screeningCombinationsPreview.length > 0 ? (
              <ol className="run-screening-plan">
                {screeningCombinationsPreview.map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ol>
            ) : (
              <p className="run-screening-dataset">
                No preview lines for this regime yet. Screening still tries mg/L (or numeric)
                predictors and numeric outputs available in the regime slice.
              </p>
            )}
            <div className="run-screening-dataset">Dataset: {preview.filename}</div>
          </div>

          {hypothesesTestedCount != null && !screeningMutation.isPending && (
            <div className="hyp-run-complete">
              <p>
                Screening finished.{" "}
                <strong>{hypothesesTestedCount.toLocaleString()}</strong> hypotheses were tested.
              </p>
              <button type="button" className="btn-primary" onClick={() => void handleViewScreeningResults()}>
                View run history and results
              </button>
              <p className="hyp-run-complete-hint">
                Opens literature-grounded screening for this regime (requires at least three corpus PDFs).
              </p>
            </div>
          )}

          <div className="step-actions" style={{ flexShrink: 0 }}>
            <button className="btn-secondary" onClick={() => setStep("configure")}>
              Back
            </button>
            <button
              className="btn-primary"
              onClick={handleLaunch}
              disabled={
                screeningMutation.isPending ||
                !segmentationReady ||
                selectedRegimeId == null
              }
            >
              {screeningMutation.isPending ? "Working…" : "Run hypothesis screening"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
