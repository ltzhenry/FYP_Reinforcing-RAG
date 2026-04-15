"""Stage 5 — Decision engine: given low confidence, decide what to do next.

Actions:
  - refine_query   : question is unclear or decomposition was poor
  - retrieve_more  : evidence is insufficient
  - regenerate     : evidence is fine but answer is weak
"""
import logging
from typing import Dict, List, Optional

from core.llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)

ACTION_REFINE = "refine_query"
ACTION_RETRIEVE = "retrieve_more"
ACTION_REGENERATE = "regenerate"


class DecisionEngine:
    def __init__(self):
        self.client, self.provider = get_llm_client("decision")
        self.model_name = get_model_name("decision", self.provider)

    def decide(self, question: str, answer: str,
               evidence: List[Dict],
               verification: Dict,
               aggregation: Dict,
               iteration: int) -> Dict:
        """Return {action, reason, extra} describing next recovery step."""

        if self.client:
            try:
                return self._llm_decide(question, answer, evidence,
                                        verification, aggregation, iteration)
            except Exception as exc:
                logger.warning("LLM decision failed: %s — using heuristic", exc)

        return self._heuristic_decide(verification, aggregation)

    # ------------------------------------------------------------------
    def _llm_decide(self, question, answer, evidence, verification, aggregation, iteration) -> Dict:
        dims = verification.get("details", {}).get("dimension_means", {})
        ctx = (
            f"Iteration {iteration}\n"
            f"Confidence: {verification['confidence']:.2f}\n"
            f"Faithfulness: {dims.get('faithfulness', 0):.2f}, "
            f"Completeness: {dims.get('completeness', 0):.2f}, "
            f"Consistency: {dims.get('consistency', 0):.2f}\n"
            f"Evidence coverage: {aggregation.get('coverage', 0):.2f}\n"
            f"Evidence count: {len(evidence)}\n"
            f"Question: {question}\n"
            f"Answer preview: {answer[:200]}"
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": (
                    "You are a meta-controller for a RAG system. "
                    "Given the current state, decide the SINGLE best next action.\n"
                    f"Choose one of: {ACTION_REFINE}, {ACTION_RETRIEVE}, {ACTION_REGENERATE}\n"
                    "Return JSON: {\"action\": \"...\", \"reason\": \"...\"}"
                )},
                {"role": "user", "content": ctx},
            ],
            temperature=0.2,
            **get_token_limit_kwargs(self.model_name, 200),
        )
        import json, re
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        parsed = json.loads(text)
        action = parsed.get("action", ACTION_REGENERATE)
        if action not in (ACTION_REFINE, ACTION_RETRIEVE, ACTION_REGENERATE):
            action = ACTION_REGENERATE
        return {"action": action, "reason": parsed.get("reason", ""), "extra": {}}

    @staticmethod
    def _heuristic_decide(verification: Dict, aggregation: Dict) -> Dict:
        dims = verification.get("details", {}).get("dimension_means", {})
        completeness = dims.get("completeness", 0)
        faithfulness = dims.get("faithfulness", 0)
        coverage = aggregation.get("coverage", 0)

        if coverage < 0.05 or completeness < 0.3:
            return {"action": ACTION_RETRIEVE,
                    "reason": "Evidence coverage or completeness is very low",
                    "extra": {}}

        if faithfulness < 0.4:
            return {"action": ACTION_REFINE,
                    "reason": "Low faithfulness suggests question/decomposition mismatch",
                    "extra": {}}

        return {"action": ACTION_REGENERATE,
                "reason": "Evidence seems adequate but answer quality is low",
                "extra": {}}
