import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { useAutomatedScreeningIteration, useUploadDataset } from "@/hooks/usePipeline";
import { apiErrorMessage } from "@/api/client";
import { inferDomainContext } from "@/api/endpoints";
import {
  clearNewRunDraft,
  DRAFT_CHANGED_EVENT,
  loadNewRunDraft,
  saveNewRunDraft,
} from "@/lib/newRunDraft";
import type { ColumnInfo, DatasetPreview } from "@/types";

/** Fallback column index slice when the file has no unified layout (CSV / plain Excel). */
const LEGACY_INPUT_START = 2;
const DOMAIN_CACHE_KEY = "aria:domainContext";
const LEGACY_INPUT_END = 37;
const LEGACY_OUTPUT_START = 37;

type Step = "upload" | "configure" | "clean_preview" | "hypotheses";

/** Shown in the header after a file exists (upload is implicit: drop / replace only). */
const POST_UPLOAD_STEPS: Step[] = ["configure", "clean_preview", "hypotheses"];
const STEP_PILL_LABELS_AFTER_UPLOAD = ["Configure", "Clean data", "Hypotheses"] as const;

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

  const [domainContext, setDomainContext] = useState<string | null>(null);
  const [domainLoading, setDomainLoading] = useState(false);
  const [nCorpusPapers, setNCorpusPapers] = useState<number | null>(null);

  const [progressMsgIdx, setProgressMsgIdx] = useState(0);
  const progressMessages = [
    "Scanning inputs, outputs, and column variability…",
    "Exploring variable mixes and multicollinearity patterns…",
    "Building the hypothesis space for signal discovery…",
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

  // Reset to upload screen when the sidebar × button clears the draft.
  useEffect(() => {
    const onDraftChanged = () => {
      if (loadNewRunDraft() !== null) return;
      setPreview(null);
      setStep("upload");
      setOutcome("");
      setFeatures([]);
      setRunName("");
      setThreshold(0.75);
      setHypothesesTestedCount(null);
      setScreeningRunId(null);
      setDomainContext(null);
      setNCorpusPapers(null);
      hydrateOnceRef.current = false;
    };
    window.addEventListener(DRAFT_CHANGED_EVENT, onDraftChanged);
    return () => window.removeEventListener(DRAFT_CHANGED_EVENT, onDraftChanged);
  }, []);

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

  useEffect(() => {
    if (step !== "hypotheses" || !preview) return;

    // Show cached domain immediately to avoid loading flash.
    type DomainCache = { domain_context: string; corpus_fingerprint: string; n_papers: number };
    let cachedFingerprint: string | null = null;
    try {
      const raw = localStorage.getItem(DOMAIN_CACHE_KEY);
      if (raw) {
        const cached = JSON.parse(raw) as DomainCache;
        if (cached.domain_context && cached.corpus_fingerprint) {
          setDomainContext(cached.domain_context);
          setNCorpusPapers(cached.n_papers);
          cachedFingerprint = cached.corpus_fingerprint;
        }
      }
    } catch { /* ignore */ }

    // Always call API to verify fingerprint (LLM is skipped server-side if cached).
    if (!cachedFingerprint) setDomainLoading(true);
    const colNames = preview.columns.map((c: ColumnInfo) => c.name);
    inferDomainContext(colNames)
      .then((res) => {
        setNCorpusPapers(res.n_papers);
        if (res.n_papers < 3) {
          setDomainContext(null);
          try { localStorage.removeItem(DOMAIN_CACHE_KEY); } catch { /* ignore */ }
          return;
        }
        if (res.corpus_fingerprint !== cachedFingerprint) {
          setDomainContext(res.domain_context || null);
        }
        if (res.domain_context) {
          try {
            localStorage.setItem(
              DOMAIN_CACHE_KEY,
              JSON.stringify({
                domain_context: res.domain_context,
                corpus_fingerprint: res.corpus_fingerprint,
                n_papers: res.n_papers,
              })
            );
          } catch { /* quota */ }
        }
      })
      .catch(() => { if (!cachedFingerprint) setDomainContext(null); })
      .finally(() => setDomainLoading(false));
  }, [step, preview]);

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    try {
      const p = await upload(files[0]);
      applyPreviewFromUpload(p);
      toast.success(`Loaded ${p.n_rows.toLocaleString()} rows`);
    } catch (error) {
      toast.error(apiErrorMessage(error, "Upload failed"));
    }
  }, [upload, applyPreviewFromUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected: () => toast.error("Use a CSV, TSV, XLS, or XLSX dataset."),
    accept: {
      "text/csv": [".csv"],
      "text/tab-separated-values": [".tsv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  /** Input / output column names from upload, or legacy index slice on ``preview.columns`` order. */
  const layoutInputCols = useMemo(() => {
    if (!preview?.columns?.length) return [];
    const fromUpload = preview.input_cols?.filter(Boolean) ?? [];
    if (fromUpload.length) return fromUpload;
    const names = preview.columns.map((c) => c.name);
    if (names.length <= LEGACY_INPUT_START) return [];
    return names.slice(LEGACY_INPUT_START, Math.min(LEGACY_INPUT_END, names.length));
  }, [preview]);

  const layoutOutputCols = useMemo(() => {
    if (!preview?.columns?.length) return [];
    const fromUpload = preview.output_cols?.filter(Boolean) ?? [];
    if (fromUpload.length) return fromUpload;
    const names = preview.columns.map((c) => c.name);
    if (names.length <= LEGACY_OUTPUT_START) return [];
    return names.slice(LEGACY_OUTPUT_START);
  }, [preview]);

  const columnByName = useMemo(() => {
    const m = new Map<string, ColumnInfo>();
    if (preview) for (const c of preview.columns) m.set(c.name, c);
    return m;
  }, [preview]);

  const sheetVaryingInputColumns = useMemo(() => {
    if (!layoutInputCols.length || !preview) return [];
    return layoutInputCols
      .filter((name) => (columnByName.get(name)?.n_unique ?? 0) > 1)
      .map((name) => columnByName.get(name))
      .filter((c): c is ColumnInfo => c != null);
  }, [layoutInputCols, preview, columnByName]);

  const sheetVaryingOutputColumns = useMemo(() => {
    if (!layoutOutputCols.length || !preview) return [];
    return layoutOutputCols
      .filter((name) => (columnByName.get(name)?.n_unique ?? 0) > 1)
      .map((name) => columnByName.get(name))
      .filter((c): c is ColumnInfo => c != null);
  }, [layoutOutputCols, preview, columnByName]);

  const showSignalColumnPanel = Boolean(
    layoutInputCols.length > 0 || layoutOutputCols.length > 0,
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

  const handleLaunch = async () => {
    if (!preview) {
      toast.error("Missing dataset.");
      return;
    }
    setHypothesesTestedCount(null);
    try {
      const r = await screeningMutation.mutateAsync({
        filename: preview.filename,
        run_name: runName.trim() || "Untitled screening",
        convergence_threshold: threshold,
      });
      setHypothesesTestedCount(r.hypotheses_tested);
      setScreeningRunId(r.run_id ?? null);
      toast.success("Screening iteration complete");
    } catch {
      toast.error("Screening failed");
    }
  };

  const handleViewScreeningResults = () => {
    if (!preview) {
      toast.error("Missing dataset.");
      return;
    }
    const q = new URLSearchParams({
      filename: preview.filename,
      regimeId: "1",
    });
    if (screeningRunId) q.set("runId", screeningRunId);
    navigate(`/runs/stats?${q.toString()}`, {
      state: { runName: runName.trim() || "Untitled screening" },
    });
  };

  return (
    <div className="new-run-page">
      <div className="page-header">
        <h1 className="page-title">New Run</h1>
        <div className="page-header-right">
          <div className="step-pills">
            {!preview ? (
              <div className="step-pill active">Upload</div>
            ) : (
              STEP_PILL_LABELS_AFTER_UPLOAD.map((label, i) => {
                const curIdx = POST_UPLOAD_STEPS.indexOf(step);
                const onFileHub = step === "upload";
                return (
                  <div
                    key={label}
                    className={`step-pill ${
                      onFileHub ? "" : i === curIdx ? "active" : i < curIdx ? "done" : ""
                    }`}
                  >
                    {label}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {step === "upload" && !preview && (
        <div className="new-run-upload">
          <div {...getRootProps()} className={`dropzone-large ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="dz-icon">⬆</div>
            <div className="dz-text">{uploading ? "Uploading..." : "Drop your dataset here"}</div>
            <div className="dz-hint">CSV, TSV, Excel · up to 100k rows</div>
          </div>
        </div>
      )}

      {step === "upload" && preview && (
        <div className="new-run-upload new-run-file-hub">
          <div className="dataset-bar" style={{ marginBottom: 16 }}>
            <span>{preview.filename}</span>
            <span>{preview.n_rows.toLocaleString()} rows</span>
            <span>{preview.n_cols} columns</span>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                clearNewRunDraft();
                setPreview(null);
                setOutcome("");
                setFeatures([]);
                setHypothesesTestedCount(null);
                setScreeningRunId(null);
                setStep("upload");
              }}
            >
              Clear dataset
            </button>
            <button
              type="button"
              className="btn-primary"
              style={{ marginLeft: 8 }}
              onClick={() => setStep("configure")}
            >
              Back to inputs / outputs →
            </button>
          </div>
          <p className="new-run-preview-caption" style={{ marginBottom: 12 }}>
            Replace the file below, or return to the cleaned sample to continue the run.
          </p>
          <div {...getRootProps()} className={`dropzone-large ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="dz-icon">⬆</div>
            <div className="dz-text">{uploading ? "Uploading..." : "Drop a replacement dataset"}</div>
            <div className="dz-hint">CSV, TSV, Excel · up to 100k rows</div>
          </div>
        </div>
      )}

      {step === "clean_preview" && preview && (
        <div className="new-run-review new-run-clean-preview">
          <div className="dataset-bar" style={{ marginBottom: 12 }}>
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
                setStep("upload");
              }}
            >
              Replace dataset
            </button>
          </div>
          <p className="new-run-preview-caption">
            Cleaned sample (first {Math.min(100, preview.n_rows)} row{preview.n_rows === 1 ? "" : "s"}).{" "}
          </p>
          <div className="new-run-preview-wrap">
            {preview.preview_rows && preview.preview_rows.length > 0 ? (
              <table className="dataset-sample-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {preview.columns.map((c) => (
                      <th key={c.name} title={c.name}>
                        {c.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview_rows.map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ color: "var(--text-muted)" }}>{ri + 1}</td>
                      {preview.columns.map((c) => (
                        <td key={c.name} title={row[c.name] ?? ""}>
                          {row[c.name] ?? ""}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ padding: 24, color: "var(--text-muted)", margin: 0 }}>
                No preview rows in the response. Re-upload the file to refresh, or continue if the column summary
                above looks correct.
              </p>
            )}
          </div>
          <div className="step-actions" style={{ marginTop: 16, flexShrink: 0 }}>
            <button type="button" className="btn-secondary" onClick={() => setStep("configure")}>
              Back
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={() => { setHypothesesTestedCount(null); setStep("hypotheses"); }}
            >
              Continue →
            </button>
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

          <div className="col-grid-scroll">
            <div className="regime-column-panels">
              {showSignalColumnPanel && (
                <>
                  <section className="regime-col-section" aria-labelledby="signal-inputs-heading">
                    <h3 id="signal-inputs-heading" className="regime-col-section-title">
                      Inputs
                      <span className="regime-col-section-meta"> · non-constant only</span>
                    </h3>
                    {sheetVaryingInputColumns.length > 0 ? (
                      <div className="col-grid">{sheetVaryingInputColumns.map(renderReadonlyColCard)}</div>
                    ) : (
                      <p className="regime-col-section-empty">No non-constant inputs in the designated range.</p>
                    )}
                  </section>
                  <section className="regime-col-section" aria-labelledby="signal-outputs-heading">
                    <h3 id="signal-outputs-heading" className="regime-col-section-title">
                      Outputs
                      <span className="regime-col-section-meta"> · non-constant only</span>
                    </h3>
                    {sheetVaryingOutputColumns.length > 0 ? (
                      <div className="col-grid">{sheetVaryingOutputColumns.map(renderReadonlyColCard)}</div>
                    ) : (
                      <p className="regime-col-section-empty">No non-constant outputs in the designated range.</p>
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
              onClick={() => setStep("clean_preview")}
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
            Signal discovery across <strong>all data slices</strong>. The next step is automated hypothesis screening.
          </p>

          {domainLoading && (
            <div className="domain-context-bar domain-context-loading">
              Inferring domain context…
            </div>
          )}
          {!domainLoading && domainContext && (
            <div className="domain-context-bar">
              <span className="domain-context-label">Identified domain</span>
              <span className="domain-context-value">{domainContext}</span>
            </div>
          )}
          {!domainLoading && nCorpusPapers !== null && nCorpusPapers < 3 && (
            <div className="domain-context-bar domain-context-warn">
              {nCorpusPapers === 0
                ? "No corpus papers uploaded. Upload at least 3 papers before running."
                : `Only ${nCorpusPapers} paper${nCorpusPapers === 1 ? "" : "s"} uploaded. Upload at least 3 for reliable literature grounding.`}
            </div>
          )}
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

          {hypothesesTestedCount != null && !screeningMutation.isPending && (
            <div className="hyp-run-complete">
              <p>
                Screening finished.{" "}
                <strong>{hypothesesTestedCount.toLocaleString()}</strong> hypotheses were tested.
              </p>
              <button type="button" className="btn-primary" onClick={handleViewScreeningResults}>
                View statistical results
              </button>
              <p className="hyp-run-complete-hint">
                Shows top hypotheses ranked by R². Literature grounding is available from the results page
                once corpus papers are uploaded.
              </p>
            </div>
          )}

          <div className="step-actions" style={{ flexShrink: 0 }}>
            <button className="btn-secondary" onClick={() => setStep("clean_preview")}>
              Back
            </button>
            <button
              className="btn-primary"
              type="button"
              onClick={() => void handleLaunch()}
              disabled={screeningMutation.isPending || !preview || domainLoading || !domainContext || (nCorpusPapers !== null && nCorpusPapers < 3)}
            >
              {screeningMutation.isPending ? "Working…" : "Run hypothesis screening"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
