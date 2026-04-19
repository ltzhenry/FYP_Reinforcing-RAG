"""ReasoningRAG — full iterative pipeline orchestrator.

Flow:
  1. Analyze query complexity
  2. Decompose if complex (LLM)
  3. Multi-hop retrieval
  4. Aggregate evidence → generate answer (LLM)
  5. Cross-model verification (dual judge)
  6. If confidence >= threshold → return
     Else → decision engine picks action (refine / retrieve more / regenerate)
     Loop up to MAX_ITERATIONS
  7. If still low confidence → fallback to raw LLM answer (clearly labelled)
  8. Attach full transparency chain throughout
"""
import logging
import time
from typing import Dict, List, Optional

from config import Config
from core.embedder import Embedder
from core.vector_store import VectorStore
from pipeline.query_analyzer import QueryAnalyzer
from pipeline.query_decomposer import QueryDecomposer
from pipeline.retriever import MultiHopRetriever
from pipeline.evidence_aggregator import EvidenceAggregator
from pipeline.answer_generator import AnswerGenerator
from pipeline.verifier import CrossModelVerifier
from pipeline.decision_engine import DecisionEngine, ACTION_REFINE, ACTION_RETRIEVE, ACTION_REGENERATE
from pipeline.iteration_state import IterationState

logger = logging.getLogger(__name__)


class ReasoningRAG:
    def __init__(self, config: Config = None,
                 embedder: Optional[Embedder] = None,
                 vector_store: Optional[VectorStore] = None):
        self.cfg = config or Config()

        self.embedder = embedder or Embedder(self.cfg.EMBEDDING_MODEL)
        self.vector_store = vector_store or VectorStore(self.embedder.embedding_dim)

        self.analyzer = QueryAnalyzer(self.cfg.COMPLEXITY_THRESHOLD)
        self.decomposer = QueryDecomposer(self.cfg.MAX_SUBQUERIES)
        self.retriever = MultiHopRetriever(
            self.vector_store, self.embedder,
            top_k=self.cfg.TOP_K_RETRIEVAL,
            max_hops=self.cfg.MAX_HOPS,
            similarity_threshold=self.cfg.SIMILARITY_THRESHOLD,
            keyword_top_k=self.cfg.KEYWORD_FALLBACK_TOP_K,
        )
        self.aggregator = EvidenceAggregator(self.cfg.MAX_EVIDENCE_LENGTH)
        self.generator = AnswerGenerator()
        self.verifier = CrossModelVerifier()
        self.decision_engine = DecisionEngine()

    # ==================================================================
    # Index management
    # ==================================================================
    def build_index(self, passages: List[Dict]):
        texts = [p["text"] for p in passages]
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.add_passages(embeddings, passages)

    def save_index(self, path: str):
        self.vector_store.save(path, metadata={"embedding_model": self.embedder.model_name})

    def load_index(self, path: str):
        self.vector_store.load(path)

    # ==================================================================
    # Main query entry point
    # ==================================================================
    def query(self, question: str) -> Dict:
        t0 = time.perf_counter()
        chain: List[Dict] = []          # transparency chain

        # ---- Step 1: analyse ------------------------------------------
        analysis = self.analyzer.analyze(question)
        chain.append({"step": "analysis", "detail": analysis})

        # ---- Step 2: decompose ----------------------------------------
        self.decomposer.refine_history = []
        subqueries = self.decomposer.decompose(analysis)
        chain.append({
            "step": "decomposition",
            "method": self.decomposer.last_method,
            "subqueries": [s["subquery"] for s in subqueries],
        })

        # ---- Step 3-6: iterative loop with persistent state -----------
        current_question = question
        extra_retrieval_queries: List[str] = []
        best_result: Optional[Dict] = None
        iteration = 0
        state = IterationState(original_question=question)

        for iteration in range(1, self.cfg.MAX_ITERATIONS + 1):
            # 3. Retrieve — accumulate evidence across iterations
            retrieval = self.retriever.retrieve(subqueries, extra_queries=extra_retrieval_queries)
            for sq in subqueries:
                state.record_query(sq.get("subquery", ""))
            for eq in extra_retrieval_queries:
                state.record_query(eq)
            state.record_evidence(retrieval["evidence"])

            chain.append({
                "step": f"retrieval_iter_{iteration}",
                "evidence_count": len(retrieval["evidence"]),
                "accumulated_evidence_count": len(state.accumulated_evidence),
                "known_titles_count": len(state.known_titles),
                "stats": retrieval["stats"],
            })

            # Use accumulated pool for aggregation so we never lose prior hits
            pool = {"evidence": list(state.accumulated_evidence)}

            # 4. Aggregate over the ACCUMULATED pool (not just this round)
            agg = self.aggregator.aggregate(pool["evidence"])
            gen = self.generator.generate(question, agg["selected"], analysis)
            answer_text = gen["answer"]
            chain.append({
                "step": f"generation_iter_{iteration}",
                "method": gen["method"],
                "answer_preview": answer_text[:200],
            })

            # 4.5 If no evidence at all, skip to fallback
            if not answer_text:
                logger.info("No answer generated (no evidence) — going to fallback")
                break

            # 5. Verify — always run, let the judge decide quality
            verification = self.verifier.verify(question, answer_text, agg["selected"])
            confidence = verification["confidence"]
            chain.append({
                "step": f"verification_iter_{iteration}",
                "confidence": confidence,
                "divergence": verification["divergence"],
                "judge_a": verification["judge_a"],
                "judge_b": verification["judge_b"],
            })

            best_result = self._pack_result(
                question=question,
                current_question=current_question,
                answer=answer_text,
                analysis=analysis,
                subqueries=subqueries,
                decomposition_record=self.decomposer.get_decomposition_record(),
                retrieval=retrieval,
                aggregation=agg,
                generation=gen,
                verification=verification,
                chain=chain,
                iteration=iteration,
                t0=t0,
                is_fallback=False,
            )

            # 6. Threshold check
            if confidence >= self.cfg.CONFIDENCE_THRESHOLD:
                logger.info("Confidence %.3f >= threshold at iteration %s", confidence, iteration)
                return best_result

            # 7. Decision
            decision = self.decision_engine.decide(
                current_question, answer_text, agg["selected"],
                verification, agg, iteration,
            )
            chain.append({
                "step": f"decision_iter_{iteration}",
                "action": decision["action"],
                "reason": decision["reason"],
            })
            logger.info("Iter %s conf=%.3f → action=%s: %s",
                        iteration, confidence, decision["action"], decision["reason"])

            # Record the action for future reference
            state.record_action(iteration, decision["action"])

            # Execute decision — now state-aware
            if decision["action"] == ACTION_REFINE:
                feedback = decision["reason"]
                # Give the refiner the accumulated evidence and known entities
                evidence_summary = state.evidence_summary_for_refine()
                known = list(state.known_titles) + sorted(state.known_entities)[:5]
                current_question = self.decomposer.refine_query(
                    current_question, feedback,
                    evidence_context=evidence_summary,
                    known_entities=known,
                )
                analysis = self.analyzer.analyze(current_question)
                subqueries = self.decomposer.decompose(analysis)
                extra_retrieval_queries = []
                chain.append({
                    "step": f"refine_iter_{iteration}",
                    "new_question": current_question,
                    "new_decomposition_method": self.decomposer.last_method,
                    "new_subqueries": [s["subquery"] for s in subqueries],
                    "evidence_context_used": bool(evidence_summary),
                    "known_entities_used": known[:6],
                })

            elif decision["action"] == ACTION_RETRIEVE:
                # Build an entity-aware probe instead of a lazy restatement
                probe = state.build_entity_probe_query()
                # Don't retry a probe we've already tried
                if state.has_tried_query(probe):
                    # Fall back to a rotation of entity-missing combinations
                    probe = probe + " details facts"
                extra_retrieval_queries.append(probe)
                chain.append({
                    "step": f"retrieve_more_iter_{iteration}",
                    "probe_query": probe,
                    "known_titles": sorted(state.known_titles),
                    "missing_concepts": state.missing_question_concepts(),
                })

            elif decision["action"] == ACTION_REGENERATE:
                chain.append({"step": f"regenerate_iter_{iteration}"})
                # loop will re-generate on next iteration with same evidence

        # Attach final state snapshot for transparency
        chain.append({"step": "iteration_state_final", "state": state.as_dict()})

        # ---- Exhausted iterations (or LLM died early) — fallback ------
        need_fallback = (
            best_result is None
            or best_result["verification"]["confidence"] < self.cfg.CONFIDENCE_THRESHOLD
        )
        if need_fallback:
            reason = ("no_evidence_answer_produced" if best_result is None
                      else "max_iterations_exhausted")
            logger.warning("Falling back — %s", reason)
            fallback_gen = self.generator.generate_fallback(question)
            chain.append({"step": "fallback", "reason": reason})
            best_result = self._pack_result(
                question=question,
                current_question=current_question,
                answer=fallback_gen["answer"],
                analysis=analysis,
                subqueries=subqueries,
                decomposition_record=self.decomposer.get_decomposition_record(),
                retrieval=retrieval,
                aggregation=agg,
                generation=fallback_gen,
                verification={"confidence": 0.0, "divergence": 0.0,
                              "judge_a": {}, "judge_b": {}, "details": {}},
                chain=chain,
                iteration=iteration,
                t0=t0,
                is_fallback=True,
            )

        return best_result

    # ==================================================================
    @staticmethod
    def _pack_result(*, question, current_question, answer, analysis, subqueries,
                     decomposition_record, retrieval, aggregation, generation,
                     verification, chain, iteration, t0, is_fallback) -> Dict:
        return {
            "question": question,
            "current_question": current_question,
            "answer": {
                "answer": answer,
                "confidence": verification["confidence"],
                "method": generation["method"],
                "is_fallback": is_fallback,
                "llm_calls": generation.get("llm_calls", 0),
            },
            "analysis": analysis,
            "subqueries": subqueries,
            "decomposition": decomposition_record,
            "retrieval": {
                "evidence": retrieval["evidence"],
                "stats": retrieval["stats"],
                "subquery_details": retrieval["subquery_details"],
            },
            "aggregation": {
                "selected": aggregation["selected"],
                "avg_quality": aggregation["avg_quality"],
                "coverage": aggregation["coverage"],
            },
            "verification": verification,
            "transparency_chain": chain,
            "metadata": {
                "iterations": iteration,
                "is_fallback": is_fallback,
                "runtime_seconds": time.perf_counter() - t0,
            },
        }
