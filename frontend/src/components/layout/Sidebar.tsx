import { useCallback, useEffect, useState } from "react";
import { UserButton, useUser } from "@clerk/clerk-react";
import { useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "@/store/theme";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { apiErrorMessage } from "@/api/client";
import { deleteRun, uploadDataset, uploadPaper } from "@/api/endpoints";
import { clearNewRunDraft, DRAFT_CHANGED_EVENT, loadNewRunDraft } from "@/lib/newRunDraft";

const NAV = [
  { to: "/upload", label: "▶ Run pipeline" },
  { to: "/runs", label: "📋 Run history" },
  { to: "/corpus", label: "📚 Corpus" },
];

export function Sidebar() {
  const { dark, toggle } = useTheme();
  const { user } = useUser();
  const navigate = useNavigate();
  const location = useLocation();

  const [draftPreview, setDraftPreview] = useState(() => loadNewRunDraft()?.preview ?? null);

  useEffect(() => {
    const refresh = () => setDraftPreview(loadNewRunDraft()?.preview ?? null);
    window.addEventListener(DRAFT_CHANGED_EVENT, refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener(DRAFT_CHANGED_EVENT, refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);

  const accountLine =
    user?.primaryEmailAddress?.emailAddress ??
    user?.username ??
    (user?.firstName ? `${user.firstName}${user.lastName ? ` ${user.lastName}` : ""}` : null);

  const onDropData = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    try {
      const preview = await uploadDataset(files[0]);
      toast.success(`${files[0].name} uploaded`);
      navigate("/upload", { state: { preview } });
    } catch (error) {
      toast.error(apiErrorMessage(error, "Upload failed"));
    }
  }, [navigate]);

  const onDropCorpus = useCallback(async (files: File[]) => {
    for (const f of files) {
      try {
        await uploadPaper(f);
        toast.success(`${f.name} added to corpus`);
      } catch (error) {
        toast.error(`${f.name}: ${apiErrorMessage(error, "Upload failed")}`);
      }
    }
  }, []);

  const { getRootProps: getDataProps, getInputProps: getDataInput, isDragActive: dataDrag } =
    useDropzone({
      onDrop: onDropData,
      onDropRejected: () => toast.error("Use a CSV, TSV, XLS, or XLSX dataset."),
      accept: {
        "text/csv": [".csv"],
        "text/tab-separated-values": [".tsv"],
        "application/vnd.ms-excel": [".xls"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      },
      maxFiles: 1,
    });

  const { getRootProps: getPdfProps, getInputProps: getPdfInput, isDragActive: pdfDrag } =
    useDropzone({
      onDrop: onDropCorpus,
      onDropRejected: () => toast.error("Use PDF files for the corpus."),
      accept: { "application/pdf": [".pdf"] },
    });

  return (
    <aside className="sidebar">
      <div className="sb-logo">
        <span className="sb-logo-dim">PFAS-</span>
        <span className="sb-logo-accent">ARIA</span>
      </div>

      <div className="sb-section">Experimental Data</div>
      <div {...getDataProps()} className={`upload-zone ${dataDrag ? "active" : ""}`}>
        <input {...getDataInput()} />
        <div className="uz-icon">⬆</div>
        <div className="uz-text">Upload dataset</div>
        <div className="uz-hint">CSV, Excel · up to 100k rows</div>
      </div>

      {draftPreview && (
        <div className="sb-dataset-pill">
          <div className="sb-dataset-info" onClick={() => navigate("/upload")}>
            <span className="sb-dataset-name" title={draftPreview.filename}>{draftPreview.filename}</span>
            <span className="sb-dataset-meta">{draftPreview.n_rows.toLocaleString()} rows · {draftPreview.n_cols} cols</span>
          </div>
          <button
            className="sb-dataset-delete"
            title="Remove dataset"
            onClick={async () => {
              const draft = loadNewRunDraft();
              if (draft?.screeningRunId) {
                try { await deleteRun(draft.screeningRunId); } catch { /* ignore */ }
              }
              clearNewRunDraft();
              navigate("/upload", { replace: true, state: {} });
            }}
          >
            ×
          </button>
        </div>
      )}

      <div className="sb-sep" />

      <div className="sb-section">Corpus</div>
      <div {...getPdfProps()} className={`upload-zone ${pdfDrag ? "active" : ""}`}>
        <input {...getPdfInput()} />
        <div className="uz-icon">📄</div>
        <div className="uz-text">Add papers</div>
        <div className="uz-hint">PDF · indexed on next run</div>
      </div>

      <div className="sb-sep" />

      {NAV.map(({ to, label }) => {
        const active =
          to === "/runs"
            ? location.pathname === "/runs" ||
              (location.pathname.startsWith("/runs/") && location.pathname !== "/runs/screening")
            : location.pathname.startsWith(to);
        return (
          <div key={to} className={`nav-item ${active ? "active" : ""}`} onClick={() => navigate(to)}>
            {label}
          </div>
        );
      })}

      <div className="sb-footer">
        <div className="sb-account">
          <div className="sb-account-meta">
            <span className="sb-account-heading">Profile</span>
            <span className="sb-account-email" title={accountLine ?? undefined}>
              {accountLine ?? "Signed in"}
            </span>
          </div>
          <div className="sb-user-button-wrap">
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  userButtonAvatarBox: "sb-clerk-avatar",
                  userButtonTrigger: "sb-clerk-trigger",
                },
              }}
            />
          </div>
        </div>
        <div className="toggle-row">
          <span className="toggle-lbl">{dark ? "Dark" : "Light"} mode</span>
          <label className="toggle-switch">
            <input type="checkbox" checked={dark} onChange={toggle} />
            <div className="toggle-track" />
            <div className="toggle-thumb" />
          </label>
        </div>
      </div>
    </aside>
  );
}
