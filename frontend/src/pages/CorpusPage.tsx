import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { BookOpen, FileText, Upload } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { getCorpusStats, uploadPaper } from "@/api/endpoints";

export function CorpusPage() {
  const { data: stats, refetch } = useQuery({
    queryKey: ["corpus"],
    queryFn: getCorpusStats,
  });

  const onDrop = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        try {
          await uploadPaper(file);
          toast.success(`${file.name} queued for indexing`);
        } catch {
          toast.error(`Failed to upload ${file.name}`);
        }
      }
      refetch();
    },
    [refetch]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
  });

  return (
    <div className="corpus-page">
      <header className="page-header">
        <h1 className="page-title">
          <BookOpen size={24} />
          Corpus
        </h1>
        <div className="corpus-stats-pills">
          <span>{stats?.n_papers ?? 0} papers</span>
          <span>{(stats?.n_chunks_total ?? 0).toLocaleString()} chunks</span>
        </div>
      </header>

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`dropzone dropzone--pdf ${isDragActive ? "dropzone--active" : ""}`}
      >
        <input {...getInputProps()} />
        <Upload size={24} />
        <p>Drop PDFs to add to corpus</p>
        <p className="dropzone-hint">Papers will be indexed on next pipeline run</p>
      </div>

      {/* Paper list */}
      <div className="paper-list">
        {stats?.papers.map((paper) => (
          <div key={paper.id} className="paper-card">
            <FileText size={16} className="paper-icon" />
            <div className="paper-info">
              <p className="paper-title">{paper.title ?? paper.filename}</p>
              <p className="paper-meta">
                {paper.n_chunks} chunks · {paper.n_tokens.toLocaleString()} tokens
                {paper.embedding_model && ` · ${paper.embedding_model}`}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
