import { useState } from "react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import { DataUploader, ColumnSelector } from "@/components/upload/DataUploader";
import { useStartRun } from "@/hooks/usePipeline";
import type { DatasetPreview } from "@/types";

type Step = "upload" | "configure" | "settings";

export function NewRun() {
  const navigate = useNavigate();
  const { mutateAsync: startRun, isPending } = useStartRun();

  const [step, setStep] = useState<Step>("upload");
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [outcomeVariable, setOutcomeVariable] = useState("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [runName, setRunName] = useState("");
  const [maxRounds, setMaxRounds] = useState(10);
  const [hypothesesPerRound, setHypothesesPerRound] = useState(6);
  const [convergenceThreshold, setConvergenceThreshold] = useState(0.75);

  const handlePreview = (p: DatasetPreview) => {
    setPreview(p);
    // Auto-select all numeric columns as features
    const numericCols = p.columns.filter((c) => c.is_numeric).map((c) => c.name);
    setFeatureColumns(numericCols);
    setStep("configure");
  };

  const handleLaunch = async () => {
    if (!preview || !outcomeVariable || featureColumns.length === 0) {
      toast.error("Select an outcome variable and at least one feature");
      return;
    }
    if (!runName.trim()) {
      toast.error("Enter a run name");
      return;
    }
    try {
      const { run_id } = await startRun({
        run_name: runName,
        filename: preview.filename,
        outcome_variable: outcomeVariable,
        feature_columns: featureColumns,
        exclude_columns: [],
        max_rounds: maxRounds,
        convergence_threshold: convergenceThreshold,
        hypotheses_per_round: hypothesesPerRound,
        strict_validation: false,
      });
      toast.success("Pipeline started!");
      navigate(`/runs/${run_id}`);
    } catch {
      toast.error("Failed to start run");
    }
  };

  return (
    <div className="new-run-page">
      <header className="page-header">
        <h1 className="page-title">New Run</h1>
        <div className="step-indicator">
          {["Upload", "Configure", "Settings"].map((s, i) => (
            <div
              key={s}
              className={`step-dot-label ${
                i === ["upload", "configure", "settings"].indexOf(step)
                  ? "active"
                  : i < ["upload", "configure", "settings"].indexOf(step)
                  ? "done"
                  : ""
              }`}
            >
              {s}
            </div>
          ))}
        </div>
      </header>

      {step === "upload" && <DataUploader onPreview={handlePreview} />}

      {step === "configure" && preview && (
        <div className="configure-step">
          <div className="dataset-summary">
            <span>{preview.filename}</span>
            <span>{preview.n_rows.toLocaleString()} rows</span>
            <span>{preview.n_cols} columns</span>
          </div>
          <ColumnSelector
            preview={preview}
            outcomeVariable={outcomeVariable}
            featureColumns={featureColumns}
            onOutcomeChange={setOutcomeVariable}
            onFeaturesChange={setFeatureColumns}
          />
          <div className="step-actions">
            <button className="btn-secondary" onClick={() => setStep("upload")}>
              Back
            </button>
            <button
              className="btn-primary"
              onClick={() => setStep("settings")}
              disabled={!outcomeVariable || featureColumns.length === 0}
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      {step === "settings" && (
        <div className="settings-step">
          <div className="settings-grid">
            <label className="field">
              <span>Run Name</span>
              <input
                type="text"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="e.g. PFAS UV study batch 1"
                className="input"
              />
            </label>
            <label className="field">
              <span>Max Rounds</span>
              <input
                type="number"
                min={1}
                max={20}
                value={maxRounds}
                onChange={(e) => setMaxRounds(Number(e.target.value))}
                className="input"
              />
            </label>
            <label className="field">
              <span>Hypotheses per Round</span>
              <input
                type="number"
                min={2}
                max={20}
                value={hypothesesPerRound}
                onChange={(e) => setHypothesesPerRound(Number(e.target.value))}
                className="input"
              />
            </label>
            <label className="field">
              <span>Convergence Threshold</span>
              <input
                type="number"
                min={0.5}
                max={1.0}
                step={0.05}
                value={convergenceThreshold}
                onChange={(e) => setConvergenceThreshold(Number(e.target.value))}
                className="input"
              />
            </label>
          </div>

          <div className="run-summary-box">
            <p>
              <strong>Outcome:</strong> {outcomeVariable}
            </p>
            <p>
              <strong>Features:</strong> {featureColumns.length} columns
            </p>
            <p>
              <strong>Dataset:</strong> {preview?.filename}
            </p>
          </div>

          <div className="step-actions">
            <button className="btn-secondary" onClick={() => setStep("configure")}>
              Back
            </button>
            <button
              className="btn-primary btn-launch"
              onClick={handleLaunch}
              disabled={isPending || !runName.trim()}
            >
              {isPending ? "Launching..." : "Launch Pipeline →"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
