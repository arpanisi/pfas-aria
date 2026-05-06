import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle } from "lucide-react";
import clsx from "clsx";
import toast from "react-hot-toast";
import type { ColumnInfo, DatasetPreview } from "@/types";
import { useUploadDataset } from "@/hooks/usePipeline";

interface Props {
  onPreview: (preview: DatasetPreview) => void;
}

export function DataUploader({ onPreview }: Props) {
  const { mutateAsync: upload, isPending } = useUploadDataset();

  const onDrop = useCallback(
    async (files: File[]) => {
      const file = files[0];
      if (!file) return;
      try {
        const preview = await upload(file);
        onPreview(preview);
        toast.success(`Loaded ${preview.n_rows.toLocaleString()} rows`);
      } catch {
        toast.error("Failed to upload file");
      }
    },
    [upload, onPreview]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "text/tab-separated-values": [".tsv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    maxFiles: 1,
    disabled: isPending,
  });

  return (
    <div
      {...getRootProps()}
      className={clsx("dropzone", {
        "dropzone--active": isDragActive,
        "dropzone--loading": isPending,
      })}
    >
      <input {...getInputProps()} />
      <Upload size={32} className="dropzone-icon" />
      {isPending ? (
        <p>Uploading and parsing...</p>
      ) : isDragActive ? (
        <p>Drop it here</p>
      ) : (
        <>
          <p className="dropzone-text">Drop your dataset here</p>
          <p className="dropzone-hint">CSV, TSV, or Excel · up to 100k rows</p>
        </>
      )}
    </div>
  );
}

// ── Column Selector ────────────────────────────────────────────────────────────

interface SelectorProps {
  preview: DatasetPreview;
  outcomeVariable: string;
  featureColumns: string[];
  onOutcomeChange: (col: string) => void;
  onFeaturesChange: (cols: string[]) => void;
}

export function ColumnSelector({
  preview,
  outcomeVariable,
  featureColumns,
  onOutcomeChange,
  onFeaturesChange,
}: SelectorProps) {
  const toggleFeature = (col: string) => {
    if (col === outcomeVariable) return;
    if (featureColumns.includes(col)) {
      onFeaturesChange(featureColumns.filter((c) => c !== col));
    } else {
      onFeaturesChange([...featureColumns, col]);
    }
  };

  return (
    <div className="column-selector">
      <div className="col-selector-header">
        <div>
          <h3>Select Outcome Variable</h3>
          <p className="hint">The variable the agent will try to explain</p>
        </div>
        <div>
          <h3>Select Feature Columns</h3>
          <p className="hint">
            {featureColumns.length} selected · LLM will generate hypotheses from these
          </p>
        </div>
      </div>

      <div className="column-grid">
        {preview.columns.map((col) => (
          <div key={col.name} className="col-card">
            <div className="col-card-header">
              <span className="col-name">{col.name}</span>
              <span className={clsx("col-type", { "col-type--num": col.is_numeric })}>
                {col.is_numeric ? "num" : "cat"}
              </span>
            </div>
            <div className="col-meta">
              <span>{col.n_unique} unique</span>
              <span>{col.missing_pct.toFixed(1)}% missing</span>
            </div>
            <div className="col-samples">
              {col.sample_values.slice(0, 3).map(String).join(", ")}
            </div>

            <div className="col-actions">
              <button
                className={clsx("col-btn col-btn--outcome", {
                  "col-btn--active": col.name === outcomeVariable,
                })}
                onClick={() => onOutcomeChange(col.name)}
              >
                {col.name === outcomeVariable ? <CheckCircle size={12} /> : null}
                Outcome
              </button>
              <button
                className={clsx("col-btn col-btn--feature", {
                  "col-btn--active": featureColumns.includes(col.name),
                  "col-btn--disabled": col.name === outcomeVariable,
                })}
                onClick={() => toggleFeature(col.name)}
                disabled={col.name === outcomeVariable}
              >
                Feature
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
