"""
Results Routes.
Endpoints for retrieving hypotheses, model results, and citations
for a given pipeline run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import CurrentUser
from src.api.database import get_db
from src.api.models.orm import Citation, Hypothesis, ModelResult, Run

router = APIRouter(prefix="/results", tags=["Results"])


# ── Response schemas ──────────────────────────────────────────────────────────


class HypothesisResponse(BaseModel):
    id: str
    hypothesis_id: str
    round: int
    description: str
    rationale: str | None
    primary_variables: list
    model_family: str
    priority_score: float


class ModelResultResponse(BaseModel):
    id: str
    model_type: str
    r_squared: float
    adj_r_squared: float
    n_observations: int
    coefficients: dict
    p_values: dict
    significant_variables: list
    validation_passed: bool
    match_score: float


class CitationResponse(BaseModel):
    id: str
    source: str
    title: str
    url: str | None
    similarity_score: float
    year: str | None


class RunSummaryResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    n_rounds: int
    final_match_score: float
    converged: bool
    outcome_variable: str | None
    n_hypotheses: int
    n_citations: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/{run_id}/summary", response_model=RunSummaryResponse)
async def get_run_summary(
    run_id: str,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> RunSummaryResponse:
    """Full summary of a completed run."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    n_hyp = len(
        (await db.execute(select(Hypothesis).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    return RunSummaryResponse(
        run_id=run.id,
        run_name=run.run_name,
        status=run.status,
        n_rounds=run.n_rounds_completed,
        final_match_score=run.final_match_score,
        converged=run.converged,
        outcome_variable=run.outcome_variable,
        n_hypotheses=n_hyp,
        n_citations=0,  # TODO: aggregate from citations table
    )


@router.get("/{run_id}/hypotheses", response_model=list[HypothesisResponse])
async def get_hypotheses(
    run_id: str,
    user: CurrentUser,
    round_number: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[HypothesisResponse]:
    """Get all hypotheses for a run, optionally filtered by round."""
    query = select(Hypothesis).where(Hypothesis.run_id == run_id)
    if round_number is not None:
        query = query.where(Hypothesis.round == round_number)
    query = query.order_by(Hypothesis.round, Hypothesis.priority_score.desc())

    result = await db.execute(query)
    hypotheses = result.scalars().all()

    return [
        HypothesisResponse(
            id=h.id,
            hypothesis_id=h.hypothesis_id,
            round=h.round,
            description=h.description,
            rationale=h.rationale,
            primary_variables=h.primary_variables,
            model_family=h.model_family,
            priority_score=h.priority_score,
        )
        for h in hypotheses
    ]


@router.get("/{run_id}/models", response_model=list[ModelResultResponse])
async def get_model_results(
    run_id: str,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> list[ModelResultResponse]:
    """Get all model results for a run."""
    hyp_ids = (
        (await db.execute(select(Hypothesis.id).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    if not hyp_ids:
        return []

    result = await db.execute(
        select(ModelResult)
        .where(ModelResult.hypothesis_id.in_(hyp_ids))
        .order_by(ModelResult.r_squared.desc())
    )
    models = result.scalars().all()

    return [
        ModelResultResponse(
            id=m.id,
            model_type=m.model_type,
            r_squared=m.r_squared,
            adj_r_squared=m.adj_r_squared,
            n_observations=m.n_observations,
            coefficients=m.coefficients,
            p_values=m.p_values,
            significant_variables=m.significant_variables,
            validation_passed=m.validation_passed,
            match_score=m.match_score,
        )
        for m in models
    ]


@router.get("/{run_id}/citations", response_model=list[CitationResponse])
async def get_citations(
    run_id: str,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> list[CitationResponse]:
    """Get all citations for a run, sorted by similarity score."""
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
        .limit(50)
    )
    citations = result.scalars().all()

    return [
        CitationResponse(
            id=c.id,
            source=c.source,
            title=c.title,
            url=c.url,
            similarity_score=c.similarity_score,
            year=c.year,
        )
        for c in citations
    ]
