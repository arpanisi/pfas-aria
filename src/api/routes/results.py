"""
Results Routes.
Endpoints for retrieving hypotheses, model results, validation,
citations, and convergence history for a given pipeline run.
Uses Redis cache for expensive aggregation queries.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select, text

from src.api.deps import CurrentUser, DBSession
from src.db.orm import Citation, Hypothesis, ModelResult, Run, ValidationResult
from src.db.redis_client import get_db_query, set_db_query
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/results", tags=["Results"])


# ── Response schemas ──────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    run_id: str
    run_name: str
    status: str
    n_rounds: int
    final_match_score: float
    converged: bool
    outcome_variable: str | None
    selected_features: list
    n_hypotheses: int


class HypothesisOut(BaseModel):
    id: str
    hypothesis_id: str
    round: int
    description: str
    rationale: str | None
    primary_variables: list
    model_family: str
    priority_score: float
    is_refinement: bool


class ModelResultOut(BaseModel):
    id: str
    hypothesis_id: str
    model_type: str
    r_squared: float
    adj_r_squared: float
    n_observations: int
    coefficients: dict
    p_values: dict
    significant_variables: list
    match_score: float
    validation_passed: bool


class ValidationOut(BaseModel):
    vif_passed: bool
    max_vif: float
    normality_passed: bool
    shapiro_p: float
    homoscedasticity_passed: bool
    bp_p: float
    anova_passed: bool
    anova_p: float
    cv_r2_mean: float
    effect_size_label: str | None
    overall_passed: bool
    pass_rate: float


class CitationOut(BaseModel):
    id: str
    source: str
    title: str
    url: str | None
    year: str | None
    similarity_score: float
    variable: str | None


class ConvergencePoint(BaseModel):
    round: int
    avg_match_score: float
    best_r_squared: float
    n_hypotheses: int
    n_passed: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/{run_id}/summary", response_model=RunSummary)
async def get_summary(run_id: str, user: CurrentUser, db: DBSession) -> RunSummary:
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")

    n_hyp_result = await db.execute(
        select(Hypothesis).where(Hypothesis.run_id == run_id)
    )
    n_hyp = len(n_hyp_result.scalars().all())

    return RunSummary(
        run_id=run.id,
        run_name=run.run_name,
        status=run.status,
        n_rounds=run.n_rounds_completed,
        final_match_score=run.final_match_score,
        converged=run.converged,
        outcome_variable=run.outcome_variable,
        selected_features=run.selected_features or [],
        n_hypotheses=n_hyp,
    )


@router.get("/{run_id}/hypotheses", response_model=list[HypothesisOut])
async def get_hypotheses(
    run_id: str,
    user: CurrentUser,
    db: DBSession,
    round_number: int | None = None,
) -> list[HypothesisOut]:
    """List hypotheses for a run. Optionally filter by round."""
    cache_key = f"{run_id}:hypotheses:{round_number}"
    cached = await get_db_query(cache_key)
    if cached:
        return [HypothesisOut(**h) for h in cached]

    query = select(Hypothesis).where(Hypothesis.run_id == run_id)
    if round_number is not None:
        query = query.where(Hypothesis.round == round_number)
    query = query.order_by(Hypothesis.round, Hypothesis.priority_score.desc())

    result = await db.execute(query)
    hypotheses = result.scalars().all()

    out = [
        HypothesisOut(
            id=h.id,
            hypothesis_id=h.hypothesis_id,
            round=h.round,
            description=h.description,
            rationale=h.rationale,
            primary_variables=h.primary_variables,
            model_family=h.model_family,
            priority_score=h.priority_score,
            is_refinement=h.is_refinement,
        )
        for h in hypotheses
    ]

    await set_db_query(cache_key, [o.model_dump() for o in out])
    return out


@router.get("/{run_id}/models", response_model=list[ModelResultOut])
async def get_model_results(
    run_id: str, user: CurrentUser, db: DBSession
) -> list[ModelResultOut]:
    """All model results for a run, sorted by R²."""
    hyp_ids = (
        (await db.execute(select(Hypothesis.id).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    if not hyp_ids:
        return []

    result = await db.execute(
        select(ModelResult, ValidationResult)
        .outerjoin(ValidationResult, ValidationResult.model_result_id == ModelResult.id)
        .where(ModelResult.hypothesis_id.in_(hyp_ids))
        .order_by(ModelResult.r_squared.desc())
    )
    rows = result.all()

    return [
        ModelResultOut(
            id=mr.id,
            hypothesis_id=mr.hypothesis_id,
            model_type=mr.model_type,
            r_squared=mr.r_squared,
            adj_r_squared=mr.adj_r_squared,
            n_observations=mr.n_observations,
            coefficients=mr.coefficients,
            p_values=mr.p_values,
            significant_variables=mr.significant_variables,
            match_score=mr.match_score,
            validation_passed=vr.overall_passed if vr else False,
        )
        for mr, vr in rows
    ]


@router.get("/{run_id}/citations", response_model=list[CitationOut])
async def get_citations(
    run_id: str, user: CurrentUser, db: DBSession
) -> list[CitationOut]:
    """All citations for a run, sorted by similarity score."""
    hyp_ids = (
        (await db.execute(select(Hypothesis.id).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    if not hyp_ids:
        return []

    model_ids = (
        (
            await db.execute(
                select(ModelResult.id).where(ModelResult.hypothesis_id.in_(hyp_ids))
            )
        )
        .scalars()
        .all()
    )

    if not model_ids:
        return []

    result = await db.execute(
        select(Citation)
        .where(Citation.model_result_id.in_(model_ids))
        .order_by(Citation.similarity_score.desc())
        .limit(100)
    )

    return [
        CitationOut(
            id=c.id,
            source=c.source,
            title=c.title,
            url=c.url,
            year=c.year,
            similarity_score=c.similarity_score,
            variable=c.variable,
        )
        for c in result.scalars().all()
    ]


@router.get("/{run_id}/convergence", response_model=list[ConvergencePoint])
async def get_convergence(
    run_id: str, user: CurrentUser, db: DBSession
) -> list[ConvergencePoint]:
    """
    Convergence score history per round.
    Reads from mv_run_convergence materialized view if available,
    falls back to live query.
    """
    try:
        result = await db.execute(
            text(
                "SELECT round, avg_match_score, best_r_squared, "
                "n_hypotheses, n_passed "
                "FROM mv_run_convergence WHERE run_id = :run_id "
                "ORDER BY round"
            ),
            {"run_id": run_id},
        )
        rows = result.all()
        if rows:
            return [
                ConvergencePoint(
                    round=r.round,
                    avg_match_score=r.avg_match_score,
                    best_r_squared=r.best_r_squared,
                    n_hypotheses=r.n_hypotheses,
                    n_passed=r.n_passed,
                )
                for r in rows
            ]
    except Exception as e:
        # A failed statement aborts the transaction in PostgreSQL; must rollback
        # before running the fallback queries on the same session.
        logger.debug("mv_run_convergence read failed, using live query: %s", e)
        await db.rollback()

    # Fallback: live aggregation query
    hyp_ids = (
        (await db.execute(select(Hypothesis.id).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    if not hyp_ids:
        return []

    result = await db.execute(
        select(Hypothesis.round, ModelResult.r_squared, ModelResult.match_score)
        .join(ModelResult, ModelResult.hypothesis_id == Hypothesis.id)
        .where(Hypothesis.run_id == run_id)
        .order_by(Hypothesis.round)
    )
    rows = result.all()

    # Aggregate by round
    rounds: dict[int, dict] = {}
    for r in rows:
        rnd = r.round
        if rnd not in rounds:
            rounds[rnd] = {"scores": [], "r2s": [], "count": 0}
        rounds[rnd]["scores"].append(r.match_score)
        rounds[rnd]["r2s"].append(r.r_squared)
        rounds[rnd]["count"] += 1

    return [
        ConvergencePoint(
            round=rnd,
            avg_match_score=sum(v["scores"]) / len(v["scores"]),
            best_r_squared=max(v["r2s"]),
            n_hypotheses=v["count"],
            n_passed=0,
        )
        for rnd, v in sorted(rounds.items())
    ]


# ── Report download ───────────────────────────────────────────────────────────


@router.get("/{run_id}/report/markdown")
async def download_markdown_report(
    run_id: str,
    user: CurrentUser,
) -> object:
    """Download the Markdown convergence report for a run."""
    from src.utils.paths import REPORTS_DIR

    path = REPORTS_DIR / f"{run_id}_report.md"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not generated yet")
    return FileResponse(
        path=str(path),
        media_type="text/markdown",
        filename=f"pfas_aria_{run_id}_report.md",
    )


@router.get("/{run_id}/report/pdf")
async def download_pdf_report(
    run_id: str,
    user: CurrentUser,
) -> object:
    """Download the PDF convergence report for a run."""
    from src.utils.paths import REPORTS_DIR

    path = REPORTS_DIR / f"{run_id}_report.pdf"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF report not generated yet")
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=f"pfas_aria_{run_id}_report.pdf",
    )
