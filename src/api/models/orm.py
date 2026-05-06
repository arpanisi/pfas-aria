"""
ORM Models.
SQLAlchemy models matching the PostgreSQL schema defined in the system design.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="initializing")
    config_version: Mapped[str] = mapped_column(String(100), nullable=True)
    data_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    n_rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    final_match_score: Mapped[float] = mapped_column(Float, default=0.0)
    converged: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome_variable: Mapped[str] = mapped_column(String(255), nullable=True)
    selected_features: Mapped[list] = mapped_column(JSON, default=list)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    regimes: Mapped[list[Regime]] = relationship(back_populates="run")
    hypotheses: Mapped[list[Hypothesis]] = relationship(back_populates="run")


class Regime(Base):
    __tablename__ = "regimes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    label: Mapped[str] = mapped_column(String(100))
    method: Mapped[str] = mapped_column(String(100))
    size: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    summary_stats: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[Run] = relationship(back_populates="regimes")


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    hypothesis_id: Mapped[str] = mapped_column(String(50))
    round: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text, nullable=True)
    primary_variables: Mapped[list] = mapped_column(JSON, default=list)
    model_family: Mapped[str] = mapped_column(String(100))
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)

    run: Mapped[Run] = relationship(back_populates="hypotheses")
    model_results: Mapped[list[ModelResult]] = relationship(back_populates="hypothesis")


class ModelResult(Base):
    __tablename__ = "model_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    hypothesis_id: Mapped[str] = mapped_column(
        ForeignKey("hypotheses.id"), nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(100))
    r_squared: Mapped[float] = mapped_column(Float, default=0.0)
    adj_r_squared: Mapped[float] = mapped_column(Float, default=0.0)
    aic: Mapped[float] = mapped_column(Float, nullable=True)
    bic: Mapped[float] = mapped_column(Float, nullable=True)
    n_observations: Mapped[int] = mapped_column(Integer, default=0)
    coefficients: Mapped[dict] = mapped_column(JSON, default=dict)
    p_values: Mapped[dict] = mapped_column(JSON, default=dict)
    significant_variables: Mapped[list] = mapped_column(JSON, default=list)
    validation_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    match_score: Mapped[float] = mapped_column(Float, default=0.0)

    hypothesis: Mapped[Hypothesis] = relationship(back_populates="model_results")
    citations: Mapped[list[Citation]] = relationship(back_populates="model_result")


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_result_id: Mapped[str] = mapped_column(
        ForeignKey("model_results.id"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    year: Mapped[str] = mapped_column(String(10), nullable=True)

    model_result: Mapped[ModelResult] = relationship(back_populates="citations")
