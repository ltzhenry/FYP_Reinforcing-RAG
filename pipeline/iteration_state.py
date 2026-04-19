"""Stateful memory carried across iterations of the reasoning loop.

Each iteration contributes observations (retrieved evidence, entities seen,
actions tried). The next iteration uses this memory to:
  - avoid re-retrieving the same queries
  - construct targeted retrieve_more queries using known vs missing entities
  - give refine_query access to evidence it has seen so far
  - prevent decision engine from looping on the same action
"""
import re
from typing import Dict, List, Set, Tuple


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "to", "for",
    "on", "at", "by", "with", "from", "as", "and", "or", "but", "does", "do",
    "did", "has", "have", "had", "be", "been", "being", "that", "this",
    "these", "those", "which", "who", "what", "how", "when", "where", "why",
    "both", "each", "same", "other", "more", "less", "also", "than",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_CAPITAL_PHRASE_RE = re.compile(r"(?:[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*)")


def _content_words(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)
            if w.lower() not in _STOPWORDS and len(w) > 1]


def _capitalized_entities(text: str) -> List[str]:
    """Cheap entity extraction: multi-word Capitalized phrases."""
    out = []
    for match in _CAPITAL_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        # filter single common words accidentally capitalized at sentence start
        if phrase.lower() in _STOPWORDS:
            continue
        if len(phrase) <= 2:
            continue
        out.append(phrase)
    return out


class IterationState:
    """Per-query memory across iterations."""

    def __init__(self, original_question: str):
        self.original_question = original_question
        self.question_entities: List[str] = _capitalized_entities(original_question)
        self.question_content_words: Set[str] = set(_content_words(original_question))

        self.evidence_ids: Set = set()               # passage ids already seen
        self.accumulated_evidence: List[Dict] = []   # all unique evidence
        self.known_titles: Set[str] = set()          # titles seen so far
        self.known_entities: Set[str] = set()        # entities found in evidence
        self.tried_queries: Set[str] = set()         # queries already sent to retriever
        self.action_history: List[Tuple[int, str]] = []  # [(iter, action), ...]

    # ------------------------------------------------------------------
    # Evidence accumulation
    # ------------------------------------------------------------------
    def record_evidence(self, evidence: List[Dict]):
        """Merge new evidence into the persistent pool, updating entity memory."""
        for e in evidence:
            pid = e.get("id", hash(e.get("text", "")))
            if pid in self.evidence_ids:
                continue
            self.evidence_ids.add(pid)
            self.accumulated_evidence.append(e)

            title = e.get("title", "")
            if title:
                self.known_titles.add(title)

            text = e.get("text", "") or ""
            for ent in _capitalized_entities(text):
                # only keep entities that were NOT in the original question
                # (we already have those) and that look meaningful
                if ent.lower() not in self.question_content_words:
                    self.known_entities.add(ent)

    def record_query(self, query: str):
        if query:
            self.tried_queries.add(query.strip().lower())

    def record_action(self, iteration: int, action: str):
        self.action_history.append((iteration, action))

    # ------------------------------------------------------------------
    # Introspection used by actions
    # ------------------------------------------------------------------
    def missing_question_concepts(self) -> List[str]:
        """Content words from the question that have NOT appeared in evidence."""
        ev_text = " ".join(e.get("text", "") for e in self.accumulated_evidence).lower()
        return [w for w in self.question_content_words if w not in ev_text]

    def recent_actions(self, n: int = 2) -> List[str]:
        return [a for _, a in self.action_history[-n:]]

    def has_tried_query(self, query: str) -> bool:
        return query and query.strip().lower() in self.tried_queries

    # ------------------------------------------------------------------
    # Query construction helpers
    # ------------------------------------------------------------------
    def build_entity_probe_query(self) -> str:
        """Construct a targeted retrieval query using known entities and
        missing question concepts. Used by retrieve_more.

        Example:
          Question: "What is the population of the city that Munsonville
                     is in the northwest corner of?"
          Known titles after iter 1: ["Munsonville", "Keene, New Hampshire"]
          Missing concepts: ["population"]
          → "population Keene New Hampshire"
        """
        missing = self.missing_question_concepts()
        known = list(self.known_titles) + sorted(self.known_entities)

        if known and missing:
            # prioritize the newest 2-3 known entities with missing concepts
            chosen_known = known[-3:]
            return " ".join(missing + chosen_known)
        if known:
            # re-probe with known entities only
            return " ".join(known[-3:])
        if missing:
            return " ".join(missing + [self.original_question])
        # last resort: original question
        return self.original_question

    def evidence_summary_for_refine(self, max_chars: int = 600) -> str:
        """Compact evidence digest used by refine_query to ground rewrites."""
        if not self.accumulated_evidence:
            return "(no evidence retrieved yet)"
        parts = []
        total = 0
        for e in self.accumulated_evidence[:4]:
            title = e.get("title", "") or "untitled"
            # first sentence of evidence body
            text = e.get("text", "").split("\n")[-1]
            snippet = text[:180].strip()
            piece = f"- {title}: {snippet}"
            if total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "\n".join(parts) if parts else "(no evidence content)"

    def as_dict(self) -> Dict:
        """Serialize for transparency chain / debugging."""
        return {
            "known_titles": sorted(self.known_titles),
            "known_entities": sorted(self.known_entities)[:20],
            "tried_query_count": len(self.tried_queries),
            "action_history": [f"iter{i}:{a}" for i, a in self.action_history],
            "evidence_pool_size": len(self.accumulated_evidence),
            "missing_concepts": self.missing_question_concepts(),
        }
