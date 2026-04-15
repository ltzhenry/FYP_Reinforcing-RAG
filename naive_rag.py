"""Naive RAG baseline — single-shot retrieve-then-generate, no reasoning loop.

This serves as the control group for benchmark comparison:
  - No query analysis / decomposition
  - Single-hop dense retrieval (top-k)
  - No evidence aggregation scoring
  - Single LLM generation call
  - No verification / decision / iteration
"""
import logging
import time
from typing import Dict, List, Optional

from config import Config
from core.embedder import Embedder
from core.vector_store import VectorStore
from pipeline.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class NaiveRAG:
    def __init__(self, config: Config = None,
                 embedder: Optional[Embedder] = None,
                 vector_store: Optional[VectorStore] = None):
        self.cfg = config or Config()
        self.embedder = embedder or Embedder(self.cfg.EMBEDDING_MODEL)
        self.vector_store = vector_store or VectorStore(self.embedder.embedding_dim)
        self.generator = AnswerGenerator()

    # ---- index management (shared interface with ReasoningRAG) --------
    def build_index(self, passages: List[Dict]):
        texts = [p["text"] for p in passages]
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.add_passages(embeddings, passages)

    def save_index(self, path: str):
        self.vector_store.save(path, metadata={"embedding_model": self.embedder.model_name})

    def load_index(self, path: str):
        self.vector_store.load(path)

    # ---- query --------------------------------------------------------
    def query(self, question: str) -> Dict:
        t0 = time.perf_counter()

        query_emb = self.embedder.embed_single(question)
        results = self.vector_store.search(
            query_emb,
            top_k=self.cfg.TOP_K_RETRIEVAL,
            query_text=question,
            keyword_top_k=self.cfg.KEYWORD_FALLBACK_TOP_K,
        )

        evidence = []
        for passage, similarity in results:
            evidence.append({
                "text": passage["text"],
                "similarity": similarity,
                "hop": 1,
                "source_query": question,
                "source": passage.get("source", "unknown"),
                "title": passage.get("title", ""),
                "quality_score": similarity,
            })

        gen = self.generator.generate(question, evidence)
        answer_text = gen["answer"]
        is_fallback = False

        if not answer_text or gen["method"] in ("llm_unanswerable", "no_evidence"):
            fb = self.generator.generate_fallback(question)
            answer_text = fb["answer"]
            gen = fb
            is_fallback = True

        sims = [e["similarity"] for e in evidence]
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        return {
            "question": question,
            "current_question": question,
            "answer": {
                "answer": answer_text,
                "confidence": avg_sim,
                "method": gen["method"],
                "is_fallback": is_fallback,
                "llm_calls": gen.get("llm_calls", 0),
            },
            "analysis": {
                "question": question,
                "complexity_score": 0.0,
                "is_complex": False,
                "query_type": "naive",
            },
            "subqueries": [{"subquery": question, "type": "direct", "order": 1}],
            "decomposition": {
                "method": "none",
                "subqueries": [],
                "refine_history": [],
            },
            "retrieval": {
                "evidence": evidence,
                "stats": {
                    "total_retrievals": 1,
                    "hops_total": 1,
                    "successful": 1 if evidence else 0,
                    "avg_similarity": avg_sim,
                    "high_quality": sum(1 for s in sims if s >= self.cfg.SIMILARITY_THRESHOLD),
                },
                "subquery_details": [],
            },
            "aggregation": {
                "selected": evidence,
                "avg_quality": avg_sim,
                "coverage": 0.0,
            },
            "verification": {
                "confidence": avg_sim,
                "divergence": 0.0,
                "judge_a": {},
                "judge_b": {},
                "details": {},
            },
            "transparency_chain": [
                {"step": "single_retrieval", "evidence_count": len(evidence)},
                {"step": "single_generation", "method": gen["method"]},
            ],
            "metadata": {
                "iterations": 1,
                "is_fallback": is_fallback,
                "runtime_seconds": time.perf_counter() - t0,
            },
        }
