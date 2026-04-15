"""Stage 1a — Analyse incoming question complexity."""
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

COMPLEX_KEYWORDS = [
    "how", "why", "explain", "describe", "compare", "contrast",
    "relationship", "mechanism", "process", "multiple", "various",
    "different", "affect", "influence", "cause", "effect",
]
CONJUNCTIONS = ["and", "or", "but", "as well as", "in addition to"]


class QueryAnalyzer:
    def __init__(self, complexity_threshold: float = 0.2):
        self.complexity_threshold = complexity_threshold

    def analyze(self, question: str) -> Dict:
        q = question.lower()
        score = 0.0
        features = {
            "has_complex_keywords": False,
            "has_multiple_clauses": False,
            "has_conjunctions": False,
            "question_length": len(question.split()),
            "is_multi_part": False,
        }

        if any(kw in q for kw in COMPLEX_KEYWORDS):
            features["has_complex_keywords"] = True
            score += 0.2

        if any(c in q for c in CONJUNCTIONS):
            features["has_conjunctions"] = True
            score += 0.15

        wc = len(question.split())
        if wc > 15:
            score += 0.2
        elif wc > 10:
            score += 0.1

        if len(re.split(r"[,;]", question)) > 2:
            features["has_multiple_clauses"] = True
            score += 0.15

        if question.count("?") > 1 or any(m in q for m in ["first", "second", "also", "additionally"]):
            features["is_multi_part"] = True
            score += 0.2

        score = min(score, 1.0)
        is_complex = score >= self.complexity_threshold
        query_type = self._infer_type(q, features)

        result = {
            "question": question,
            "complexity_score": score,
            "is_complex": is_complex,
            "features": features,
            "requires_decomposition": is_complex,
            "query_type": query_type,
        }
        logger.info("Complexity %.2f (%s), type=%s", score, "complex" if is_complex else "simple", query_type)
        return result

    @staticmethod
    def _infer_type(q: str, f: Dict) -> str:
        if f["is_multi_part"]:
            return "multi-part"
        if any(w in q for w in ["compare", "contrast", "difference", "versus", "vs"]):
            return "comparative"
        if any(w in q for w in ["why", "cause", "reason", "effect"]):
            return "causal"
        if any(w in q for w in ["how", "mechanism", "process"]):
            return "procedural"
        if any(w in q for w in ["explain", "describe", "what is"]):
            return "descriptive"
        if f["has_conjunctions"]:
            return "compound"
        return "factual"
