"""Stage 3a — Aggregate and score retrieved evidence."""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class EvidenceAggregator:
    def __init__(self, max_evidence_length: int = 3000):
        self.max_len = max_evidence_length

    def aggregate(self, evidence: List[Dict]) -> Dict:
        if not evidence:
            return {"selected": [], "avg_quality": 0.0, "coverage": 0.0}

        scored = self._score(evidence)
        selected = self._select(scored)

        qualities = [e["quality_score"] for e in selected]
        avg_q = sum(qualities) / len(qualities) if qualities else 0.0
        coverage = self._coverage(selected)

        logger.info("Aggregated %s/%s evidence, avg_quality=%.3f, coverage=%.2f",
                     len(selected), len(evidence), avg_q, coverage)

        return {
            "selected": selected,
            "avg_quality": avg_q,
            "coverage": coverage,
        }

    # ------------------------------------------------------------------
    def _score(self, evidence: List[Dict]) -> List[Dict]:
        out = []
        for e in evidence:
            sim_w = e["similarity"] * 0.5
            hop_w = max(0, 1.0 - (e["hop"] - 1) * 0.2) * 0.3
            wc = len(e["text"].split())
            len_w = (1.0 if 30 <= wc <= 250 else (wc / 30 if wc < 30 else max(0.4, 1.0 - (wc - 250) / 500))) * 0.2
            scored = {**e, "quality_score": sim_w + hop_w + len_w}
            out.append(scored)
        out.sort(key=lambda x: x["quality_score"], reverse=True)
        return out

    def _select(self, scored: List[Dict]) -> List[Dict]:
        selected = []
        total_len = 0
        for e in scored:
            elen = len(e["text"])
            if total_len + elen > self.max_len and selected:
                break
            selected.append(e)
            total_len += elen
        return selected

    @staticmethod
    def _coverage(selected: List[Dict]) -> float:
        if len(selected) < 2:
            return 1.0 if selected else 0.0
        all_tokens: set = set()
        per_evidence = []
        for e in selected:
            tokens = set(e["text"].lower().split())
            per_evidence.append(tokens)
            all_tokens |= tokens
        if not all_tokens:
            return 0.0
        overlap = per_evidence[0]
        for t in per_evidence[1:]:
            overlap &= t
        return len(overlap) / len(all_tokens)
