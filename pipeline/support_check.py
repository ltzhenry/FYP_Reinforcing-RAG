"""Deterministic support / grounding checks used by the verifier.

These are cheap, non-LLM functions that provide an anchor signal so the
LLM-judge confidence cannot collapse the whole pipeline by itself.
"""
import re
import string
from typing import Dict, List, Sequence


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "to",
    "for", "on", "at", "by", "with", "from", "as", "and", "or",
    "do", "does", "did", "has", "have", "had", "be", "been", "being",
    "that", "this", "these", "those", "which", "who", "what", "how",
    "when", "where", "why",
}


def normalize_answer_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _content_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def is_short_answer(answer: str, max_words: int = 6) -> bool:
    """Short factoid-style answer: name, number, date, short phrase."""
    norm = normalize_answer_text(answer)
    return bool(norm) and len(norm.split()) <= max_words


def deterministic_support_check(answer: str,
                                evidence: Sequence[Dict]) -> Dict:
    """Check whether *answer* is directly supported by *evidence* using
    cheap lexical heuristics.

    Returns a dict with:
      - score (0.0–1.0): how well the answer is supported
      - match_type: which rule fired (for logging)
      - matched_tokens / total_tokens: debug info
    """
    norm_answer = normalize_answer_text(answer)
    if not norm_answer:
        return {"score": 0.0, "match_type": "empty_answer"}

    # Collect evidence text once
    evidence_text = " ".join(e.get("text", "") for e in evidence).lower()
    evidence_text_norm = normalize_answer_text(evidence_text)
    if not evidence_text_norm:
        return {"score": 0.0, "match_type": "no_evidence"}

    # Rule 1: exact substring match for short-ish answers (strongest signal)
    if len(norm_answer) <= 80 and norm_answer in evidence_text_norm:
        return {
            "score": 1.0,
            "match_type": "exact_substring",
            "matched_tokens": len(norm_answer.split()),
            "total_tokens": len(norm_answer.split()),
        }

    # Rule 2: content-token overlap ratio
    ans_tokens = _content_tokens(norm_answer)
    if not ans_tokens:
        return {"score": 0.0, "match_type": "no_content_tokens"}

    ev_tokens = set(_content_tokens(evidence_text_norm))
    matched = sum(1 for t in ans_tokens if t in ev_tokens)
    ratio = matched / len(ans_tokens)

    if ratio >= 0.85:
        score = 0.75
        mt = "strong_token_overlap"
    elif ratio >= 0.6:
        score = 0.5
        mt = "moderate_token_overlap"
    elif ratio >= 0.4:
        score = 0.3
        mt = "weak_token_overlap"
    else:
        score = 0.0
        mt = "no_overlap"

    return {
        "score": score,
        "match_type": mt,
        "matched_tokens": matched,
        "total_tokens": len(ans_tokens),
        "ratio": ratio,
    }


def aggregate_confidence(judge_a: Dict, judge_b: Dict,
                         support: Dict) -> Dict:
    """Confidence aggregation — LLM judges are primary, det_support is a
    weak auxiliary signal, NOT a high floor.

    Rules:
      - Weighted blend: 0.75 * llm + 0.25 * det_support
      - Failed judges (all-zero) are ignored when computing llm average
      - Guardrail: if both judges are near-zero, det_support alone can NOT
        push confidence past the acceptance threshold. This prevents
        substring matches from hijacking the verifier.
      - Mild divergence penalty when both judges are valid.
    """
    dims = ["faithfulness", "completeness", "consistency"]

    def judge_score(j: Dict) -> float:
        vals = [float(j.get(d, 0.0)) for d in dims]
        return sum(vals) / len(vals) if vals else 0.0

    conf_a = judge_score(judge_a)
    conf_b = judge_score(judge_b)

    FAILURE_EPS = 0.02
    valid_confs = [c for c in (conf_a, conf_b) if c > FAILURE_EPS]
    valid_count = len(valid_confs)

    if valid_confs:
        llm_confidence = sum(valid_confs) / len(valid_confs)
    else:
        llm_confidence = 0.0

    support_score = support.get("score", 0.0)

    # --- Weighted blend: LLM judges dominate, det_support helps a bit ---
    confidence = 0.75 * llm_confidence + 0.25 * support_score

    # --- Divergence penalty when both judges disagreed ---
    divergence = abs(conf_a - conf_b) if valid_count == 2 else 0.0
    if valid_count == 2:
        penalty = min(divergence * 0.2, 0.1)
        confidence = max(0.0, confidence - penalty)
    else:
        penalty = 0.0

    # --- Guardrail: substring match alone cannot auto-pass ---
    # If both judges voted near zero, they probably saw a real issue
    # (answer may just be a distractor mentioned in evidence).
    # Cap confidence so acceptance policy won't rubber-stamp it.
    BOTH_LOW = conf_a < 0.2 and conf_b < 0.2
    if BOTH_LOW and support_score >= 0.9:
        confidence = min(confidence, 0.45)

    # Means for logging / downstream
    means = {d: (judge_a.get(d, 0) + judge_b.get(d, 0)) / 2 for d in dims}

    return {
        "confidence": max(0.0, min(1.0, confidence)),
        "llm_confidence": llm_confidence,
        "deterministic_support_score": support_score,
        "divergence": divergence,
        "valid_judge_count": valid_count,
        "guardrail_capped": BOTH_LOW and support_score >= 0.9,
        "dimension_means": means,
        "dimension_divergences": {
            d: abs(judge_a.get(d, 0) - judge_b.get(d, 0)) for d in dims
        },
        "raw_confidence": llm_confidence,
        "penalty": penalty,
    }
