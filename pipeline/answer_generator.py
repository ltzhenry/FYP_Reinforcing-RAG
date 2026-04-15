"""Stage 3b — Generate an answer from aggregated evidence via LLM."""
import logging
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
            return {"answer": "No evidence available.", "method": "no_evidence", "llm_calls": 0}

        if self.client:
            try:
                answer = self._llm_generate(question, evidence)
                if answer and not self._is_refusal(answer) and len(answer.strip()) > 5:
                    return {"answer": answer, "method": "llm", "llm_calls": 1}
                if answer and self._is_refusal(answer):
                    return {"answer": answer, "method": "llm_unanswerable", "llm_calls": 1}
            except Exception as exc:
                logger.warning("LLM generation failed: %s", exc)

        answer = self._simple_generate(evidence)
        return {"answer": answer, "method": "simple_fallback", "llm_calls": 0}

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        low = answer.strip().lower()
        refusal_phrases = [
            "unanswerable", "insufficient evidence", "cannot determine",
            "not enough information", "no evidence", "cannot be determined",
            "unable to answer", "not mentioned",
        ]
        return any(phrase in low for phrase in refusal_phrases)

    def generate_fallback(self, question: str) -> Dict:
        """Fallback: answer purely from LLM knowledge, no evidence."""
        if not self.client:
            return {
                "answer": "Unable to answer — no evidence found and LLM unavailable.",
                "method": "no_llm_fallback", "llm_calls": 0,
            }
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
            return {"answer": "Unable to answer this question.", "method": "error", "llm_calls": 0}

    # ------------------------------------------------------------------
    def _llm_generate(self, question: str, evidence: List[Dict]) -> str:
        ctx = "\n\n".join(
            f"[{i}] {e['text']}" for i, e in enumerate(evidence[:6], 1)
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": (
                    "Extract the answer from the evidence. "
                    "Output ONLY the answer itself — a name, number, date, or short phrase. "
                    "No sentences, no explanation, no preamble.\n"
                    "Examples of GOOD answers: 'Paris', '1755', 'yes', 'Kurt Vonnegut', 'Dutch'\n"
                    "Examples of BAD answers: 'The answer is Paris.', 'Based on the evidence, it was founded in 1755.'\n"
                    "If the evidence does not contain the answer, output exactly: unanswerable"
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
        """Strip common LLM wrapper patterns to get just the answer span."""
        import re
        cleaned = raw.strip().strip('"').strip("'").strip("*").strip()
        cleaned = re.sub(r"^(the answer is|answer:|a:|based on.*?,)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip(".")
        return cleaned.strip() if cleaned.strip() else raw

    @staticmethod
    def _simple_generate(evidence: List[Dict]) -> str:
        top = evidence[0]["text"]
        sentences = [s.strip() for s in top.split(".") if s.strip()]
        return sentences[0] + "." if sentences else top[:200]
