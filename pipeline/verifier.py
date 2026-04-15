"""Stage 4 — Cross-model verification: two LLM judges score
evidence-answer consistency, then fuse into a confidence score."""
import logging
from typing import Dict, List, Optional

from core.llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an impartial judge evaluating a QA system's answer.

Question: {question}

Evidence:
{evidence_block}

Answer: {answer}

Score on 0.0–1.0:
1. **Faithfulness** — Is every claim in the answer supported by the evidence? (If the answer says "unanswerable" or "insufficient evidence", faithfulness=1.0 only if the evidence truly lacks the info.)
2. **Completeness** — Does the answer actually provide the information the question asks for? (IMPORTANT: If the answer is a refusal like "unanswerable", "insufficient evidence", or "I cannot determine", completeness MUST be 0.0–0.2 because the user's question is not answered.)
3. **Consistency** — Are there contradictions between the answer and evidence?

Return ONLY a JSON object: {{"faithfulness": 0.0, "completeness": 0.0, "consistency": 0.0}}
"""


class CrossModelVerifier:
    """Use two different LLMs (judge_a and judge_b) to independently score
    an answer, then fuse the scores into a single confidence value."""

    def __init__(self):
        self.client_a, self.prov_a = get_llm_client("judge_a")
        self.model_a = get_model_name("judge_a", self.prov_a)
        self.client_b, self.prov_b = get_llm_client("judge_b")
        self.model_b = get_model_name("judge_b", self.prov_b)

    def verify(self, question: str, answer: str,
               evidence: List[Dict]) -> Dict:
        evidence_block = "\n\n".join(
            f"[{i}] {e['text'][:500]}" for i, e in enumerate(evidence[:5], 1)
        )
        prompt = JUDGE_PROMPT.format(
            question=question,
            evidence_block=evidence_block,
            answer=answer,
        )

        score_a = self._judge(self.client_a, self.model_a, prompt, "judge_a")
        score_b = self._judge(self.client_b, self.model_b, prompt, "judge_b")

        fused = self._fuse(score_a, score_b)

        return {
            "judge_a": score_a,
            "judge_b": score_b,
            "confidence": fused["confidence"],
            "divergence": fused["divergence"],
            "details": fused,
        }

    # ------------------------------------------------------------------
    def _judge(self, client, model: Optional[str], prompt: str, label: str) -> Dict:
        defaults = {"faithfulness": 0.0, "completeness": 0.0, "consistency": 0.0}
        if not client or not model:
            logger.warning("%s unavailable — returning zeros", label)
            return defaults
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                **get_token_limit_kwargs(model, 200),
            )
            import json, re
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text).strip()
            parsed = json.loads(text)
            return {
                "faithfulness": float(parsed.get("faithfulness", 0)),
                "completeness": float(parsed.get("completeness", 0)),
                "consistency": float(parsed.get("consistency", 0)),
            }
        except Exception as exc:
            logger.warning("%s scoring failed: %s", label, exc)
            return defaults

    @staticmethod
    def _fuse(a: Dict, b: Dict) -> Dict:
        dims = ["faithfulness", "completeness", "consistency"]
        means = {d: (a[d] + b[d]) / 2 for d in dims}
        divergences = {d: abs(a[d] - b[d]) for d in dims}
        avg_divergence = sum(divergences.values()) / len(dims)

        raw_confidence = sum(means.values()) / len(dims)
        penalty = min(avg_divergence * 0.3, 0.15)
        confidence = max(0.0, min(1.0, raw_confidence - penalty))

        return {
            "confidence": confidence,
            "divergence": avg_divergence,
            "dimension_means": means,
            "dimension_divergences": divergences,
            "raw_confidence": raw_confidence,
            "penalty": penalty,
        }
