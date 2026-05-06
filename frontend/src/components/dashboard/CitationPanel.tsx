import { BookOpen, ExternalLink } from "lucide-react";
import type { Citation } from "@/types";

interface Props {
  citations: Citation[];
}

const SOURCE_LABEL: Record<string, string> = {
  corpus: "Your Papers",
  arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar",
};

export function CitationPanel({ citations }: Props) {
  const top = citations.slice(0, 15);

  return (
    <div className="card">
      <h3 className="card-title">
        <BookOpen size={16} />
        Literature Citations ({citations.length})
      </h3>
      <div className="citation-list">
        {top.map((c) => (
          <div key={c.id} className="citation-item">
            <div className="citation-header">
              <span className={`source-badge source-badge--${c.source}`}>
                {SOURCE_LABEL[c.source] ?? c.source}
              </span>
              {c.year && <span className="citation-year">{c.year}</span>}
              <span className="citation-score">
                {(c.similarity_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="citation-title">{c.title}</p>
            {c.variable && (
              <span className="citation-var">↳ {c.variable}</span>
            )}
            {c.url && (
              <a
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-link"
              >
                <ExternalLink size={12} />
                View paper
              </a>
            )}
          </div>
        ))}
        {citations.length === 0 && (
          <p className="empty-state">No citations yet</p>
        )}
      </div>
    </div>
  );
}
