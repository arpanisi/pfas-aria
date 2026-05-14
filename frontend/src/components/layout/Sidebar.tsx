import { useCallback } from "react";
import { UserButton, useUser } from "@clerk/clerk-react";
import { useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "@/store/theme";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { uploadDataset, uploadPaper } from "@/api/endpoints";

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
    } catch {
      toast.error("Upload failed");
    }
  }, [navigate]);

  const onDropCorpus = useCallback(async (files: File[]) => {
    for (const f of files) {
      try {
        await uploadPaper(f);
        toast.success(`${f.name} added to corpus`);
      } catch {
        toast.error(`Failed: ${f.name}`);
      }
    }
  }, []);

  const { getRootProps: getDataProps, getInputProps: getDataInput, isDragActive: dataDrag } =
    useDropzone({ onDrop: onDropData, accept: { "text/csv": [".csv"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] }, maxFiles: 1 });

  const { getRootProps: getPdfProps, getInputProps: getPdfInput, isDragActive: pdfDrag } =
    useDropzone({ onDrop: onDropCorpus, accept: { "application/pdf": [".pdf"] } });

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
