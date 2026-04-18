"""Stage 3b — Generate an answer from aggregated evidence via LLM.

Design principle: the generator ALWAYS produces an answer. It never refuses.
Quality judgment is the verifier's job, not the generator's.
"""
import logging
import re
from typing import Dict, List, Optional

from core.llm_provider import get_llm_client, get_model_name, get_token_limit_kwargs

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self):
        self.client, self.provider = get_llm_client("generation")
        self.model_name = get_model_name("generation", self.provider)

    def generate(self, question: str, evidence: List[Dict],
                 analysis: Optional[Dict] = None) -> Dict:
        if not evidence:
            return {"answer": "", "method": "no_evidence", "llm_calls": 0}

        if self.client:
            try:
                answer = self._llm_generate(question, evidence)
                if answer and len(answer.strip()) > 1:
                    return {"answer": answer, "method": "llm", "llm_calls": 1}
            except Exception as exc:
                logger.warning("LLM generation failed: %s", exc)

        answer = self._simple_generate(evidence)
        return {"answer": answer, "method": "simple_fallback", "llm_calls": 0}

    def generate_fallback(self, question: str) -> Dict:
        """Fallback: answer purely from LLM knowledge, no evidence."""
        if not self.client:
            return {"answer": "", "method": "no_llm_fallback", "llm_calls": 0}
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": (
                        "Answer with ONLY the answer itself — a name, number, date, or short phrase. "
                        "No explanation. Use your general knowledge."
                    )},
                    {"role": "user", "content": f"Q: {question}\nA:"},
                ],
                temperature=0.0,
                **get_token_limit_kwargs(self.model_name, 30),
            )
            return {
                "answer": self._clean_answer(resp.choices[0].message.content.strip()),
                "method": "llm_fallback_no_evidence",
                "llm_calls": 1,
            }
        except Exception as exc:
            logger.warning("Fallback generation failed: %s", exc)
            return {"answer": "", "method": "error", "llm_calls": 0}

    # ------------------------------------------------------------------
    def _llm_generate(self, question: str, evidence: List[Dict]) -> str:
        ctx = "\n\n".join(
            f"[{i}] {e['text']}" for i, e in enumerate(evidence[:6], 1)
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": (
                    "You are a QA extraction system. You MUST output an answer.\n"
                    "Output ONLY the answer — a name, number, date, or short phrase.\n"
                    "No sentences, no explanation, no preamble.\n"
                    "Examples: 'Paris', '1755', 'yes', 'Kurt Vonnegut', 'Dutch'\n\n"
                    "Rules:\n"
                    "- Use the evidence to find or reason about the answer.\n"
                    "- If multiple evidence pieces are needed, connect them.\n"
                    "- If the evidence is only partially relevant, use what you can and make your best inference.\n"
                    "- You are FORBIDDEN from saying 'unanswerable', 'unknown', 'not enough information', etc.\n"
                    "- You MUST always output a concrete answer, even if uncertain."
                )},
                {"role": "user", "content": f"{ctx}\n\nQ: {question}\nA:"},
            ],
            temperature=0.0,
            **get_token_limit_kwargs(self.model_name, 30),
        )
        raw = resp.choices[0].message.content.strip()
        return self._clean_answer(raw)

    @staticmethod
    def _clean_answer(raw: str) -> str:
        cleaned = raw.strip().strip('"').strip("'").strip("*").strip()
        cleaned = re.sub(r"^(the answer is|answer:|a:|based on.*?,)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip(".")
        return cleaned.strip() if cleaned.strip() else raw

    @staticmethod
    def _simple_generate(evidence: List[Dict]) -> str:
        text = evidence[0]["text"]
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        body_lines = [ln for ln in lines if not ln.startswith("Title:")]
        body = " ".join(body_lines) if body_lines else " ".join(lines)
        sentences = [s.strip() for s in body.split(".") if s.strip() and len(s.strip()) > 5]
        if sentences:
            return sentences[0]
        return body[:150]
