"""Evaluation metrics for the Reasoning RAG pipeline."""
import logging
import re
import string
from collections import Counter
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def evaluate_batch(self, results: List[Dict], ground_truth: List[Dict],
                       system_name: str = "ReasoningRAG") -> Dict:
        if not results:
            return {"system_name": system_name, "total_queries": 0}

        em_scores, f1_scores, coverages = [], [], []
        confidences, iterations_list, latencies = [], [], []
        fallback_count = 0

        for result, gt in zip(results, ground_truth):
            pred = result["answer"]["answer"]
            refs = gt.get("answers", [])

            em_scores.append(self._exact_match(pred, refs))
            f1_scores.append(self._token_f1(pred, refs))
            coverages.append(self._answer_coverage(pred, refs))

            confidences.append(result["answer"].get("confidence", 0))
            iterations_list.append(result["metadata"].get("iterations", 1))
            latencies.append(result["metadata"].get("runtime_seconds", 0))
            if result["metadata"].get("is_fallback"):
                fallback_count += 1

        n = len(results)
        return {
            "system_name": system_name,
            "total_queries": n,
            "exact_match": float(np.mean(em_scores)),
            "token_f1": float(np.mean(f1_scores)),
            "answer_coverage": float(np.mean(coverages)),
            "avg_confidence": float(np.mean(confidences)),
            "avg_iterations": float(np.mean(iterations_list)),
            "avg_latency_seconds": float(np.mean(latencies)),
            "fallback_rate": fallback_count / n,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        return " ".join(text.split())

    def _exact_match(self, pred: str, refs: List[str]) -> float:
        p = self._normalize(pred)
        return float(any(self._normalize(r) == p for r in refs)) if refs else 0.0

    def _token_f1(self, pred: str, refs: List[str]) -> float:
        if not refs:
            return 0.0
        pred_tokens = self._normalize(pred).split()
        best = 0.0
        for ref in refs:
            ref_tokens = self._normalize(ref).split()
            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())
            if num_common == 0:
                continue
            prec = num_common / len(pred_tokens) if pred_tokens else 0
            rec = num_common / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            best = max(best, f1)
        return best

    def _answer_coverage(self, pred: str, refs: List[str]) -> float:
        if not refs:
            return 0.0
        pred_tokens = set(self._normalize(pred).split())
        best = 0.0
        for ref in refs:
            ref_tokens = set(self._normalize(ref).split())
            if ref_tokens:
                best = max(best, len(pred_tokens & ref_tokens) / len(ref_tokens))
        return best

    def print_summary(self, metrics: Dict):
        logger.info("=== %s ===", metrics.get("system_name", ""))
        for k in ["exact_match", "token_f1", "answer_coverage",
                   "avg_confidence", "avg_iterations", "avg_latency_seconds", "fallback_rate"]:
            logger.info("  %-25s %s", k, f"{metrics.get(k, 0):.4f}")
