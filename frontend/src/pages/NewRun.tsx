import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { useUploadDataset, useStartRun } from "@/hooks/usePipeline";
import {
  clearNewRunDraft,
  loadNewRunDraft,
  saveNewRunDraft,
} from "@/lib/newRunDraft";
import type { DatasetPreview } from "@/types";

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
    const nums = p.columns.filter((c) => c.is_numeric).map((c) => c.name);
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

  const toggleFeature = (col: string) => {
    if (col === outcome) return;
    setFeatures((f) => f.includes(col) ? f.filter((c) => c !== col) : [...f, col]);
  };

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

  const steps: Step[] = ["upload", "configure", "settings"];

  return (
    <div className="new-run-page">
      <div className="page-header">
        <h1 className="page-title">New Run</h1>
        <div className="step-pills">
          {["Upload", "Configure", "Settings"].map((s, i) => {
            const key = steps[i];
            const cur = steps.indexOf(step);
            return (
              <div key={s} className={`step-pill ${i === cur ? "active" : i < cur ? "done" : ""}`}>{s}</div>
            );
          })}
        </div>
      </div>

      {step === "upload" && (
        <div className="new-run-upload">
          {preview && (
            <div className="dataset-bar" style={{ marginBottom: 16 }}>
              <span>{preview.filename}</span>
              <span>{preview.n_rows.toLocaleString()} rows</span>
              <span>{preview.n_cols} columns</span>
              {preview.excel_header_row != null && preview.excel_header_row > 0 && (
                <span style={{ color: "var(--text-muted)" }}>
                  Excel: header row {preview.excel_header_row + 1}
                  {" "}(skipped {preview.excel_header_row} preamble row
                  {preview.excel_header_row === 1 ? "" : "s"})
                </span>
              )}
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
            {preview.excel_header_row != null && preview.excel_header_row > 0 && (
              <span style={{ color: "var(--text-muted)" }}>
                Excel header row {preview.excel_header_row + 1}
              </span>
            )}
            <span style={{ marginLeft: "auto", color: "var(--text-dim)" }}>
              {features.length} features selected · outcome: {outcome || "none"}
            </span>
          </div>
          <div className="col-grid-scroll">
            <div className="col-grid">
            {preview.columns.map((col) => (
              <div key={col.name} className="col-card">
                <div className="col-card-header">
                  <span className="col-name">{col.name}</span>
                  <span className={`col-dtype ${col.is_numeric ? "num" : ""}`}>
                    {col.is_numeric ? "num" : "cat"}
                  </span>
                </div>
                <div className="col-meta">{col.n_unique} unique · {col.missing_pct.toFixed(1)}% missing</div>
                <div className="col-sample">{col.sample_values.slice(0, 3).map(String).join(", ")}</div>
                <div className="col-actions">
                  <button
                    className={`col-btn outcome ${col.name === outcome ? "active" : ""}`}
                    onClick={() => setOutcome(col.name)}
                  >Outcome</button>
                  <button
                    className={`col-btn feature ${features.includes(col.name) ? "active" : ""}`}
                    onClick={() => toggleFeature(col.name)}
                    disabled={col.name === outcome}
                  >Feature</button>
                </div>
              </div>
            ))}
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
            {preview.excel_header_row != null && preview.excel_header_row > 0 && (
              <span style={{ color: "var(--text-muted)" }}>
                Excel header row {preview.excel_header_row + 1}
              </span>
            )}
            <span style={{ marginLeft: "auto", color: "var(--text-dim)" }}>
              Outcome: {outcome} · {features.length} features
            </span>
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
