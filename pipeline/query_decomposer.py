"""Stage 1b — Decompose complex questions into sub-queries via LLM."""
import json
import logging
import re
from typing import Dict, List

from core.llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)


class QueryDecomposer:
    def __init__(self, max_subqueries: int = 4):
        self.max_subqueries = max_subqueries
        self.client, self.provider = get_llm_client("decomposition")
        self.model_name = get_model_name("decomposition", self.provider)

        self.last_method: str = ""
        self.last_subqueries: List[Dict] = []
        self.refine_history: List[Dict] = []

    def decompose(self, analysis: Dict) -> List[Dict]:
        question = analysis["question"]
        if not analysis["requires_decomposition"]:
            self.last_method = "direct"
            self.last_subqueries = [self._direct(question)]
            return self.last_subqueries

        if self.client:
            try:
                subs = self._llm_decompose(question, analysis)
                if subs:
                    self.last_method = "llm"
                    self.last_subqueries = subs
                    return subs
            except Exception as exc:
                logger.warning("LLM decomposition failed: %s — falling back to rules", exc)

        result = self._rule_decompose(question, analysis)
        self.last_method = "rule_fallback"
        self.last_subqueries = result
        return result

    def refine_query(self, question: str, feedback: str,
                     evidence_context: str = "",
                     known_entities: List[str] = None) -> str:
        """Rewrite the question using LLM with awareness of what we've
        already retrieved.

        Parameters:
            question: original or current question
            feedback: reason from decision engine
            evidence_context: a compact summary of evidence retrieved so far
            known_entities: names/titles already known from evidence
        """
        if not self.client:
            self.refine_history.append({
                "original": question, "refined": question,
                "feedback": feedback, "method": "no_llm",
            })
            return question

        ent_str = ", ".join((known_entities or [])[:6]) or "(none)"

        user_prompt = (
            f"Original question: {question}\n\n"
            f"Why we need to refine: {feedback}\n\n"
            f"Evidence already retrieved:\n{evidence_context or '(none)'}\n\n"
            f"Known entities from evidence: {ent_str}\n\n"
            "Rewrite the question so that it explicitly incorporates the "
            "KNOWN entities as anchors and asks specifically for the "
            "missing fact. The rewrite should help retrieve the missing "
            "passage. Return only the rewritten question."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": (
                         "You rewrite questions to improve retrieval. "
                         "Anchor the rewrite on entities already known, "
                         "and make the missing fact explicit."
                     )},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                **get_token_limit_kwargs(self.model_name, 200),
            )
            refined = resp.choices[0].message.content.strip()
            refined = refined if refined else question
            self.refine_history.append({
                "original": question, "refined": refined,
                "feedback": feedback,
                "known_entities": list(known_entities or [])[:6],
                "method": "llm_evidence_aware",
            })
            return refined
        except Exception as exc:
            logger.warning("Query refinement failed: %s", exc)
            self.refine_history.append({
                "original": question, "refined": question,
                "feedback": feedback, "method": "error", "error": str(exc),
            })
            return question

    def get_decomposition_record(self) -> Dict:
        """Snapshot of the last decomposition for transparency reporting."""
        return {
            "method": self.last_method,
            "subqueries": [
                {"order": s["order"], "type": s["type"], "subquery": s["subquery"]}
                for s in self.last_subqueries
            ],
            "refine_history": list(self.refine_history),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _direct(q: str) -> Dict:
        return {"subquery": q, "type": "direct", "order": 1, "dependency": None}

    def _llm_decompose(self, question: str, analysis: Dict) -> List[Dict]:
        prompt = (
            "Break this question into 2-4 simpler sub-questions for a retrieval system.\n"
            f"Question: {question}\n"
            "Return ONLY a JSON array: [{\"subquery\": \"...\", \"type\": \"...\", \"order\": N}]"
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            **get_token_limit_kwargs(self.model_name, 500),
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        raw = json.loads(text)

        out = []
        for i, sq in enumerate(raw[: self.max_subqueries], 1):
            out.append({
                "subquery": sq["subquery"],
                "type": sq.get("type", "llm_generated"),
                "order": i,
                "dependency": None if i == 1 else i - 1,
            })
        return out

    def _rule_decompose(self, question: str, analysis: Dict) -> List[Dict]:
        subs: List[Dict] = []
        if analysis["features"]["has_conjunctions"]:
            for pattern in [r"\band\b", r"\bor\b", r"\bas well as\b"]:
                parts = re.split(pattern, question, flags=re.IGNORECASE)
                if len(parts) > 1:
                    for i, p in enumerate(parts, 1):
                        p = p.strip()
                        if p and len(p) > 5:
                            subs.append({"subquery": p if p.endswith("?") else p + "?",
                                         "type": "conjunction_split", "order": i,
                                         "dependency": None if i == 1 else i - 1})
                    break

        if not subs and analysis["features"]["has_multiple_clauses"]:
            parts = re.split(r"[,;]", question)
            for i, p in enumerate(parts, 1):
                p = p.strip()
                if p and len(p) > 5:
                    subs.append({"subquery": p if p.endswith("?") else p + "?",
                                 "type": "punctuation_split", "order": i,
                                 "dependency": None if i == 1 else i - 1})

        if not subs:
            subs = [self._direct(question)]

        return subs[: self.max_subqueries]
