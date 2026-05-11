import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { clearCorpus, deletePaper, getCorpusStats, uploadPaper } from "@/api/endpoints";

export function CorpusPage() {
  const queryClient = useQueryClient();
  const { data: stats, refetch } = useQuery({ queryKey: ["corpus"], queryFn: getCorpusStats });

  const onDrop = useCallback(async (files: File[]) => {
    for (const f of files) {
      try { const result = await uploadPaper(f); toast.success(`${f.name}: ${result.status}`); }
      catch { toast.error(`Failed: ${f.name}`); }
    }
    refetch();
  }, [refetch]);

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
      refetch();
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

      <div {...getRootProps()} className={`dropzone-large ${isDragActive ? "active" : ""}`} style={{ padding: "32px 24px" }}>
        <input {...getInputProps()} />
        <div className="dz-icon">📄</div>
        <div className="dz-text">Drop PDFs to add to corpus</div>
        <div className="dz-hint">Indexed immediately into the corpus</div>
      </div>

      <div className="paper-list">
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
