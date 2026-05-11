import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { getLegacySegmentationPreview } from "@/api/endpoints";
import { useUploadDataset, useStartRun } from "@/hooks/usePipeline";
import {
  clearNewRunDraft,
  loadNewRunDraft,
  saveNewRunDraft,
} from "@/lib/newRunDraft";
import type { ColumnInfo, DatasetPreview, LegacyRegimeSummary } from "@/types";

type Step = "upload" | "configure" | "settings";

type UploadLocationState = { preview?: DatasetPreview };

export function NewRun() {
  const navigate = useNavigate();
  const location = useLocation();
  const { mutateAsync: upload, isPending: uploading } = useUploadDataset();
  const { mutateAsync: startRun, isPending: launching } = useStartRun();

  const [step, setStep] = useState<Step>("upload");
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [outcome, setOutcome] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [runName, setRunName] = useState("");
  const [maxRounds, setMaxRounds] = useState(10);
  const [hypPerRound, setHypPerRound] = useState(6);
  const [threshold, setThreshold] = useState(0.75);

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
    setMaxRounds(draft.maxRounds);
    setHypPerRound(draft.hypPerRound);
    setThreshold(draft.threshold);
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
      maxRounds,
      hypPerRound,
      threshold,
    });
  }, [preview, step, outcome, features, runName, maxRounds, hypPerRound, threshold]);

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
    enabled: step === "configure" && !!preview?.filename,
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

  const steps: Step[] = ["upload", "configure", "settings"];

  const handleLaunch = async () => {
    if (!preview || !outcome || !features.length || !runName.trim()) {
      toast.error("Fill in all required fields"); return;
    }
    try {
      const { run_id } = await startRun({
        run_name: runName, filename: preview.filename,
        outcome_variable: outcome, feature_columns: features,
        exclude_columns: [], max_rounds: maxRounds,
        convergence_threshold: threshold, hypotheses_per_round: hypPerRound,
        strict_validation: false,
      });
      toast.success("Pipeline started!");
      clearNewRunDraft();
      navigate(`/runs/${run_id}`);
    } catch { toast.error("Failed to start run"); }
  };

  return (
    <div className="new-run-page">
      <div className="page-header">
        <h1 className="page-title">New Run</h1>
        <div className="page-header-right">
          {preview && step === "configure" && (
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
            {["Upload", "Configure", "Settings"].map((s, i) => {
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
                {segQuery.data.warnings.length > 0 && (
                  <div className="seg-preview-warn">
                    {segQuery.data.warnings.join(" ")}
                  </div>
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
            <button className="btn-primary" onClick={() => setStep("settings")} disabled={!outcome || !features.length}>Continue →</button>
          </div>
        </div>
      )}

      {step === "settings" && preview && (
        <div className="new-run-settings">
          <div className="dataset-bar" style={{ marginBottom: 16 }}>
            <span>{preview.filename}</span>
            <span>{preview.n_rows.toLocaleString()} rows</span>
            <span>{preview.n_cols} columns</span>
          </div>
          <div className="settings-grid">
            <div className="field">
              <label>Run Name</label>
              <input className="input" value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="e.g. PFAS UV Batch 1" />
            </div>
            <div className="field">
              <label>Max Rounds</label>
              <input className="input" type="number" min={1} max={20} value={maxRounds} onChange={(e) => setMaxRounds(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Hypotheses per Round</label>
              <input className="input" type="number" min={2} max={20} value={hypPerRound} onChange={(e) => setHypPerRound(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Convergence Threshold</label>
              <input className="input" type="number" min={0.5} max={1.0} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            </div>
          </div>
          <div className="run-summary-box">
            <span><strong>Outcome:</strong> {outcome}</span>
            <span><strong>Features:</strong> {features.length} columns selected</span>
            <span><strong>Dataset:</strong> {preview?.filename}</span>
          </div>
          <div className="step-actions" style={{ flexShrink: 0 }}>
            <button className="btn-secondary" onClick={() => setStep("configure")}>Back</button>
            <button className="btn-primary" onClick={handleLaunch} disabled={launching || !runName.trim()}>
              {launching ? "Launching..." : "Launch Pipeline →"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
