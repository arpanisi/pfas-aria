import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import toast from "react-hot-toast";
import { clearCorpus, deletePaper, getCorpusStats, uploadPaper } from "@/api/endpoints";

export function CorpusPage() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const needPapers = searchParams.get("needPapers") === "1";
  const { data: stats } = useQuery({ queryKey: ["corpus"], queryFn: getCorpusStats });
  const [uploading, setUploading] = useState<Record<string, number>>({});

  const onDrop = useCallback(async (files: File[]) => {
    const initial: Record<string, number> = {};
    files.forEach((f) => { initial[f.name] = 0; });
    setUploading(initial);

    for (const f of files) {
      try {
        const result = await uploadPaper(f, (pct) =>
          setUploading((prev) => ({ ...prev, [f.name]: pct }))
        );
        toast.success(`${f.name}: ${result.status}`);
        await queryClient.invalidateQueries({ queryKey: ["corpus"] });
      } catch {
        toast.error(`Failed to upload: ${f.name}`);
      } finally {
        setUploading((prev) => { const next = { ...prev }; delete next[f.name]; return next; });
      }
    }
  }, [queryClient]);

  const onDeletePaper = async (paperId: string) => {
    try {
      await deletePaper(paperId);
      await queryClient.invalidateQueries({ queryKey: ["corpus"] });
      toast.success("Paper deleted");
    } catch (err) {
      console.error("Failed to delete paper", err);
      toast.error("Failed to delete paper");
    }
  };
  
  const onClearCorpus = async () => {
    const confirmed = window.confirm(
      "Clear the entire corpus? This deletes all papers and chunks."
    );
    if (!confirmed) return;
    try {
      const result = await clearCorpus();
      toast.success(`Cleared ${result.deleted_papers} papers`);
      await queryClient.invalidateQueries({ queryKey: ["corpus"] });
    } catch (err) {
      console.error("Failed to clear corpus", err);
      toast.error("Failed to clear corpus");
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "application/pdf": [".pdf"] },
  });

  return (
    <div className="corpus-page">
      <div className="page-header">
        <h1 className="page-title">Corpus</h1>
        <button className="btn danger" onClick={onClearCorpus}>Clear corpus</button>
        <div className="corpus-stats">
          <span className="stat-pill">{stats?.n_papers ?? 0} papers</span>
          <span className="stat-pill">{(stats?.n_chunks_total ?? 0).toLocaleString()} chunks</span>
          <span className="stat-pill">{(stats?.n_tokens_total ?? 0).toLocaleString()} tokens</span>
        </div>
      </div>

      {needPapers && (
        <p className="hyp-run-lead" style={{ marginBottom: 16 }}>
          Literature-grounded screening needs at least <strong>three</strong> uploaded PDFs. Add papers below, then
          return to <strong>New Run</strong> and open &quot;View run history and results&quot; again.
        </p>
      )}

      <div {...getRootProps()} className={`dropzone-large ${isDragActive ? "active" : ""}`} style={{ padding: "32px 24px" }}>
        <input {...getInputProps()} />
        <div className="dz-icon">📄</div>
        <div className="dz-text">Drop PDFs to add to corpus</div>
        <div className="dz-hint">Indexed immediately into the corpus</div>
      </div>

      <div className="paper-list">
        {Object.entries(uploading).map(([name, pct]) => (
          <div key={name} className="paper-card uploading">
            <span className="paper-icon" aria-hidden>⏳</span>
            <div className="paper-card-body">
              <div className="paper-title">{name}</div>
              <div className="paper-upload-progress">
                <div className="paper-upload-bar" style={{ width: `${pct}%` }} />
              </div>
              <div className="paper-meta">{pct < 100 ? `Uploading ${pct}%` : "Processing…"}</div>
            </div>
          </div>
        ))}
        {stats?.papers.map((p) => (
          <div key={p.id} className="paper-card">
            <span className="paper-icon" aria-hidden>
              📄
            </span>
            <div className="paper-card-body">
              <div className="paper-title">{p.title ?? p.filename}</div>
              <div className="paper-meta">
                {p.n_chunks} chunks · {p.n_tokens.toLocaleString()} tokens
                {p.embedding_model && ` · ${p.embedding_model}`}
              </div>
            </div>
            <button
              type="button"
              className="paper-delete"
              title="Delete paper"
              aria-label={`Remove ${p.filename}`}
              onClick={(e) => {
                e.stopPropagation();
                void onDeletePaper(p.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
