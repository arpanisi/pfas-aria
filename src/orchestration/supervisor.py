"""
Supervisor Agent.
Orchestrates all specialist agents using LangGraph StateGraph.
Routes between agents, maintains state, and decides when to stop.

Graph structure:
  ingest → build_rag → analyze_data → [loop start]
    → generate_hypotheses → model → ground → check_convergence
      → STOP (if converged) or → generate_hypotheses (next round)
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.orchestration.state import PipelineState, PipelineStatus
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisorAgent:
    """
    Builds and runs the LangGraph pipeline graph.
    Each node is one agent or pipeline step.
    Edges define the flow including the convergence loop.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph."""
        graph = StateGraph(dict)

        # Add all nodes
        graph.add_node("ingest", self._node_ingest)
        graph.add_node("build_rag", self._node_build_rag)
        graph.add_node("analyze_data", self._node_analyze_data)
        graph.add_node("generate_hypotheses", self._node_generate_hypotheses)
        graph.add_node("model", self._node_model)
        graph.add_node("ground", self._node_ground)
        graph.add_node("check_convergence", self._node_check_convergence)
        graph.add_node("finalize", self._node_finalize)

        # Define edges
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "build_rag")
        graph.add_edge("build_rag", "analyze_data")
        graph.add_edge("analyze_data", "generate_hypotheses")
        graph.add_edge("generate_hypotheses", "model")
        graph.add_edge("model", "ground")
        graph.add_edge("ground", "check_convergence")

        # Conditional edge: loop or stop
        graph.add_conditional_edges(
            "check_convergence",
            self._route_after_convergence_check,
            {
                "continue": "generate_hypotheses",
                "stop": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    def _route_after_convergence_check(self, state: dict) -> str:
        """Decide whether to loop or stop."""
        pipeline_state: PipelineState = state["pipeline_state"]
        if pipeline_state.should_stop:
            return "stop"
        return "continue"

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_ingest(self, state: dict) -> dict:
        """Load data and corpus."""
        ps: PipelineState = state["pipeline_state"]
        ps.status = PipelineStatus.INGESTING
        logger.info("── Node: Ingest ──")

        try:
            from src.ingestion.corpus_loader import CorpusLoader
            from src.ingestion.data_loader import DataLoader

            ps.data_bundle = DataLoader().load()
            ps.corpus_bundle = CorpusLoader().load()
            logger.info(
                f"Ingested {ps.data_bundle.n_rows} rows, "
                f"{ps.corpus_bundle.n_papers} papers"
            )
        except Exception as e:
            ps.errors.append(f"Ingestion failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"Ingestion failed: {e}")

        return {"pipeline_state": ps}

    def _node_build_rag(self, state: dict) -> dict:
        """Build the RAG vector index."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        ps.status = PipelineStatus.BUILDING_RAG
        logger.info("── Node: Build RAG ──")

        try:
            from src.rag.pipeline import RAGPipeline

            rag = RAGPipeline()
            ps.retriever = rag.build()
            logger.info(f"RAG index ready: {ps.retriever.stats()}")
        except Exception as e:
            ps.errors.append(f"RAG build failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"RAG build failed: {e}")

        return {"pipeline_state": ps}

    def _node_analyze_data(self, state: dict) -> dict:
        """Run Data Intelligence Agent."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        ps.status = PipelineStatus.ANALYZING_DATA
        logger.info("── Node: Analyze Data ──")

        try:
            from src.agents.data_intelligence import DataIntelligenceAgent

            agent = DataIntelligenceAgent()
            assert ps.data_bundle is not None, "data_bundle not loaded"
            ps.intelligence_report = agent.run(ps.data_bundle)
            logger.info(
                f"Regimes: {ps.intelligence_report.n_regimes}, "
                f"Top features: {ps.intelligence_report.top_features[:5]}"
            )
        except Exception as e:
            ps.errors.append(f"Data analysis failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"Data analysis failed: {e}")

        return {"pipeline_state": ps}

    def _node_generate_hypotheses(self, state: dict) -> dict:
        """Run Hypothesis Agent."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        ps.status = PipelineStatus.GENERATING_HYPOTHESES
        ps.current_round += 1
        logger.info(f"── Node: Generate Hypotheses (Round {ps.current_round}) ──")

        try:
            from src.agents.hypothesis import HypothesisAgent

            agent = HypothesisAgent()
            previous_results = (
                ps.all_structured_results if ps.current_round > 1 else None
            )
            assert ps.intelligence_report is not None, "intelligence_report not set"
            assert ps.retriever is not None, "retriever not set"
            ps.hypothesis_set = agent.run(
                report=ps.intelligence_report,
                retriever=ps.retriever,
                round_number=ps.current_round,
                previous_results=previous_results,
            )
            logger.info(f"Generated {ps.hypothesis_set.n_hypotheses} hypotheses")
        except Exception as e:
            ps.errors.append(f"Hypothesis generation failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"Hypothesis generation failed: {e}")

        return {"pipeline_state": ps}

    def _node_model(self, state: dict) -> dict:
        """Run Modeling + Validation Engine."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        ps.status = PipelineStatus.MODELING
        logger.info(f"── Node: Model (Round {ps.current_round}) ──")

        try:
            from src.modeling.runner import ModelingRunner

            runner = ModelingRunner()
            assert ps.hypothesis_set is not None, "hypothesis_set not set"
            assert ps.intelligence_report is not None, "intelligence_report not set"
            ps.modeling_result = runner.run(
                hypothesis_set=ps.hypothesis_set,
                df=ps.intelligence_report.labeled_df,
                per_regime=True,
            )
            logger.info(
                f"Modeling: {ps.modeling_result.n_passed}/"
                f"{ps.modeling_result.n_hypotheses} passed, "
                f"best R²={ps.modeling_result.best_r_squared:.4f}"
            )
        except Exception as e:
            ps.errors.append(f"Modeling failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"Modeling failed: {e}")

        return {"pipeline_state": ps}

    def _node_ground(self, state: dict) -> dict:
        """Run Literature Grounding Agent."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        ps.status = PipelineStatus.GROUNDING
        logger.info(f"── Node: Ground (Round {ps.current_round}) ──")

        try:
            from src.grounding.agent import GroundingAgent

            assert ps.retriever is not None, "retriever not set"
            agent = GroundingAgent(retriever=ps.retriever)
            assert ps.modeling_result is not None, "modeling_result not set"
            ps.grounding_result = agent.run(
                modeling_result=ps.modeling_result,
                convergence_threshold=ps.convergence_threshold,
            )
            logger.info(
                f"Grounding: match={ps.grounding_result.global_match_score:.4f}, "
                f"citations={ps.grounding_result.n_citations_found}"
            )
        except Exception as e:
            ps.errors.append(f"Grounding failed: {e}")
            ps.status = PipelineStatus.FAILED
            logger.error(f"Grounding failed: {e}")

        return {"pipeline_state": ps}

    def _node_check_convergence(self, state: dict) -> dict:
        """Update state with round summary and check convergence."""
        ps: PipelineState = state["pipeline_state"]
        if ps.status == PipelineStatus.FAILED:
            return {"pipeline_state": ps}

        logger.info(f"── Node: Check Convergence (Round {ps.current_round}) ──")

        ps.add_round_summary(
            grounding_result=ps.grounding_result,
            modeling_result=ps.modeling_result,
        )

        latest_score = ps.convergence_scores[-1]
        ps.final_match_score = latest_score

        logger.info(
            f"Round {ps.current_round} complete — "
            f"match={latest_score:.4f}, "
            f"stop={ps.should_stop} ({ps.stop_reason})"
        )

        if ps.should_stop:
            ps.status = (
                PipelineStatus.CONVERGED
                if latest_score >= ps.convergence_threshold
                else PipelineStatus.MAX_ROUNDS_REACHED
            )

        return {"pipeline_state": ps}

    def _node_finalize(self, state: dict) -> dict:
        """Final node — pipeline complete."""
        ps: PipelineState = state["pipeline_state"]
        logger.info(
            f"── Pipeline Complete ── "
            f"Status: {ps.status}, "
            f"Rounds: {ps.current_round}, "
            f"Final score: {ps.final_match_score:.4f}"
        )
        return {"pipeline_state": ps}

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, run_name: str | None = None) -> PipelineState:
        """Execute the full pipeline and return the final state."""
        cfg = self.settings

        ps = PipelineState(
            run_id=str(uuid.uuid4())[:8],
            run_name=run_name or f"run_{uuid.uuid4().hex[:6]}",
            max_rounds=cfg.supervisor.max_rounds,
            convergence_threshold=cfg.supervisor.convergence_threshold,
        )

        logger.info(
            f"╔══════════════════════════════════════╗\n"
            f"  PFAS-ARIA Pipeline Starting\n"
            f"  Run ID: {ps.run_id}\n"
            f"  Max rounds: {ps.max_rounds}\n"
            f"  Convergence threshold: {ps.convergence_threshold}\n"
            f"╚══════════════════════════════════════╝"
        )

        final_state = self._graph.invoke({"pipeline_state": ps})
        result: PipelineState = final_state["pipeline_state"]
        return result
