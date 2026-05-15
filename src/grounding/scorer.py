"""
Grounding Scorer.
Computes match scores between model findings and literature.

Three components:
  1. RAG similarity  — cosine similarity against your corpus (MongoDB-stored embeddings)
  2. arXiv score     — best match against fetched arXiv papers
  3. Validation rate — passed through from ValidationReport

Final match_score = weighted average per convergence config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.grounding.arxiv_client import ArxivPaper
from src.grounding.semantic_scholar import S2Paper
from src.modeling.engine import ModelResult
from src.rag.embedder import get_embedder
from src.rag.retriever import Retriever
from src.utils.config import get_settings
from src.utils.logging import get_logger
from src.validation.validator import ValidationReport

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class Citation:
    """A single grounded citation for a model finding."""

    source: str  # "corpus" | "arxiv" | "semantic_scholar"
    title: str
    authors: list[str]
    url: str
    year: str | None
    similarity_score: float
    abstract_snippet: str  # First 300 chars of abstract


@dataclass
class FindingScore:
    """Grounding result for one model finding (one significant variable)."""

    variable: str
    coefficient: float
    finding_text: str  # Human-readable description of the finding
    rag_similarity: float
    arxiv_score: float
    validation_pass_rate: float
    match_score: float  # Weighted composite
    citations: list[Citation] = field(default_factory=list)


@dataclass
class GroundingResult:
    """Full grounding output for one hypothesis."""

    hypothesis_id: str
    finding_scores: list[FindingScore]
    global_match_score: float  # Mean across all findings
    top_citations: list[Citation]  # Deduplicated, sorted by score
    all_citations: list[Citation]
    grounding_summary: str  # Plain-language summary


# ── Scorer ────────────────────────────────────────────────────────────────────


class GroundingScorer:
    """
    Scores model findings against literature.
    Combines RAG similarity, arXiv search, and validation pass rate
    into a single match_score per finding.
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.embedder = get_embedder()
        self.settings = get_settings()
        self.weights = self.settings.convergence.weights

    def score(
        self,
        model_result: ModelResult,
        validation_report: ValidationReport,
        arxiv_papers: list[ArxivPaper],
        s2_papers: list[S2Paper],
    ) -> GroundingResult:
        """
        Score all significant findings from a model result against literature.
        """
        finding_scores: list[FindingScore] = []
        all_citations: list[Citation] = []

        # Score each significant variable as a finding
        variables_to_score = (
            model_result.significant_variables
            or list(model_result.coefficients.keys())[:5]
        )

        for var in variables_to_score:
            coeff = model_result.coefficients.get(var, 0.0)
            finding_text = self._build_finding_text(
                var, coeff, model_result.outcome_variable
            )

            # 1. RAG similarity against your 50 papers
            rag_score, rag_citations = self._score_against_rag(finding_text)

            # 2. arXiv similarity
            arxiv_score, arxiv_citations = self._score_against_papers(
                finding_text,
                [
                    (
                        p.title + " " + p.abstract,
                        p.title,
                        p.authors,
                        p.url,
                        str(p.published),
                    )
                    for p in arxiv_papers
                ],
                source="arxiv",
            )

            # 3. Semantic Scholar similarity
            _, s2_citations = self._score_against_papers(
                finding_text,
                [
                    (
                        p.title + " " + p.abstract,
                        p.title,
                        p.authors,
                        p.url,
                        str(p.year),
                    )
                    for p in s2_papers
                ],
                source="semantic_scholar",
            )

            # 4. Validation pass rate
            val_rate = validation_report.pass_rate

            # 5. Weighted composite
            match_score = (
                self.weights.rag_similarity * rag_score
                + self.weights.arxiv_citation * arxiv_score
                + self.weights.validation_pass_rate * val_rate
            )
            match_score = round(min(1.0, max(0.0, match_score)), 4)

            citations = rag_citations + arxiv_citations + s2_citations
            citations = sorted(
                citations, key=lambda c: c.similarity_score, reverse=True
            )

            finding_scores.append(
                FindingScore(
                    variable=var,
                    coefficient=coeff,
                    finding_text=finding_text,
                    rag_similarity=rag_score,
                    arxiv_score=arxiv_score,
                    validation_pass_rate=val_rate,
                    match_score=match_score,
                    citations=citations[:5],
                )
            )
            all_citations.extend(citations)

        global_match = (
            float(np.mean([fs.match_score for fs in finding_scores]))
            if finding_scores
            else 0.0
        )

        # Deduplicate and rank citations
        seen_urls: set[str] = set()
        top_citations: list[Citation] = []
        for c in sorted(all_citations, key=lambda x: x.similarity_score, reverse=True):
            if c.url not in seen_urls:
                top_citations.append(c)
                seen_urls.add(c.url)
            if len(top_citations) >= 10:
                break

        summary = self._build_summary(finding_scores, global_match)

        return GroundingResult(
            hypothesis_id=model_result.hypothesis_id,
            finding_scores=finding_scores,
            global_match_score=round(global_match, 4),
            top_citations=top_citations,
            all_citations=list({c.url: c for c in all_citations}.values()),
            grounding_summary=summary,
        )

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_against_rag(self, finding_text: str) -> tuple[float, list[Citation]]:
        """Score finding against your corpus via embedding similarity."""
        try:
            chunks = self.retriever.retrieve_for_grounding(finding_text, top_k=5)
            if not chunks:
                return 0.0, []

            best_score = chunks[0].similarity_score if chunks else 0.0
            min_sim = self.settings.grounding.min_similarity_score
            citations = [
                Citation(
                    source="corpus",
                    title=c.title,
                    authors=[],
                    url=c.source_file,
                    year=None,
                    similarity_score=c.similarity_score,
                    abstract_snippet=c.text[:300],
                )
                for c in chunks
                if c.similarity_score >= min_sim
            ]
            return float(best_score), citations
        except Exception as e:
            logger.warning(f"RAG scoring failed: {e}")
            return 0.0, []

    def _score_against_papers(
        self,
        finding_text: str,
        papers: list[tuple[str, str, list[str], str, str | None]],
        source: str,
    ) -> tuple[float, list[Citation]]:
        """
        Score finding against a list of (text, title, authors, url, year) tuples.
        Uses cosine similarity of embeddings.
        """
        if not papers:
            return 0.0, []

        try:
            finding_emb = self.embedder.embed_one(finding_text)
            texts = [p[0][:1000] for p in papers]  # Cap text length
            paper_embs = self.embedder.embed(texts)

            similarities = self.embedder.batch_similarity(finding_emb, paper_embs)
            citations = []

            for i, (text, title, authors, url, year) in enumerate(papers):
                sim = float(similarities[i])
                if sim >= 0.40:
                    citations.append(
                        Citation(
                            source=source,
                            title=title,
                            authors=authors,
                            url=url,
                            year=year,
                            similarity_score=round(sim, 4),
                            abstract_snippet=text[:300],
                        )
                    )

            citations = sorted(
                citations, key=lambda c: c.similarity_score, reverse=True
            )
            best_score = citations[0].similarity_score if citations else 0.0
            return float(best_score), citations[:5]

        except Exception as e:
            logger.warning(f"Paper similarity scoring failed: {e}")
            return 0.0, []

    def _build_finding_text(
        self, variable: str, coefficient: float, outcome: str
    ) -> str:
        """Convert a coefficient into a searchable finding statement."""
        direction = "increases" if coefficient > 0 else "decreases"
        magnitude = abs(coefficient)
        return f"{variable} {direction} {outcome} " f"(coefficient={magnitude:.4f})"

    def _build_summary(
        self, finding_scores: list[FindingScore], global_score: float
    ) -> str:
        if not finding_scores:
            return "No findings to ground."

        top = sorted(finding_scores, key=lambda fs: fs.match_score, reverse=True)[:3]
        lines = [
            f"Global match score: {global_score:.3f}",
            "Top grounded findings:",
        ]
        for fs in top:
            lines.append(
                f"  {fs.variable}: match={fs.match_score:.3f}, "
                f"citations={len(fs.citations)}"
            )
        return "\n".join(lines)
