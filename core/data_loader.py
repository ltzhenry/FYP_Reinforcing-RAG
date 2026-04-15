"""Pluggable dataset loader — HotpotQA by default, switchable at runtime."""
import logging
import random
from typing import Dict, List, Sequence

from datasets import load_dataset

logger = logging.getLogger(__name__)

HOTPOT_DATASET_ID = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"


class DataLoader:
    """Load experiment / user data and expose a unified passage + question API."""

    def __init__(self, random_seed: int = 42):
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.random_seed = random_seed
        self.dataset_label = f"HotpotQA ({HOTPOT_DATASET_ID}, {HOTPOT_CONFIG})"

    # ------------------------------------------------------------------
    # Public: load
    # ------------------------------------------------------------------
    def load_dataset(self, dataset_id: str = HOTPOT_DATASET_ID,
                     config: str = HOTPOT_CONFIG, train_ratio: float = 0.8):
        logger.info("Loading dataset %s / %s …", dataset_id, config)
        self.dataset_label = f"{dataset_id} ({config})"

        try:
            ds = load_dataset(dataset_id, config)
            available = list(ds.keys())
            logger.info("Available splits: %s", available)

            if "train" in ds and "validation" in ds:
                self.train_data = ds["train"]
                self.test_data = ds["validation"]
            else:
                main = ds[available[0]]
                indices = list(range(len(main)))
                random.seed(self.random_seed)
                random.shuffle(indices)
                sp = int(len(indices) * train_ratio)
                self.train_data = [main[i] for i in indices[:sp]]
                self.test_data = [main[i] for i in indices[sp:]]

            self.dataset = ds
            logger.info("Train: %s  |  Test/Val: %s", len(self.train_data), len(self.test_data))

        except Exception as exc:
            logger.error("Dataset loading failed: %s", exc)
            self.train_data, self.test_data = [], []

    # ------------------------------------------------------------------
    # Extract context passages for indexing (HotpotQA-aware)
    # ------------------------------------------------------------------
    def get_passages(self, split: str = "train", max_passages: int = None) -> List[Dict]:
        data = self.train_data if split == "train" else self.test_data
        if not data:
            return []

        passages: List[Dict] = []
        pid = 0
        seen = set()

        for item in data:
            for raw in self._extract_passages(item):
                if raw["text"] in seen:
                    continue
                seen.add(raw["text"])
                passages.append({"id": pid, **raw})
                pid += 1
                if max_passages and len(passages) >= max_passages:
                    logger.info("Reached max_passages=%s", max_passages)
                    return passages

        logger.info("Extracted %s passages from '%s'", len(passages), split)
        return passages

    # ------------------------------------------------------------------
    # Extract evaluation questions
    # ------------------------------------------------------------------
    def get_questions(self, split: str = "test", max_questions: int = None) -> List[Dict]:
        data = self.train_data if split == "train" else self.test_data
        if not data:
            return []

        questions = []
        for item in data:
            q = item.get("question", "")
            if not q:
                continue
            a = item.get("answer", "")
            sf = item.get("supporting_facts") or {}
            questions.append({
                "id": item.get("id", ""),
                "question": q,
                "answers": [a] if a else [],
                "question_type": item.get("type"),
                "difficulty": item.get("level"),
                "supporting_titles": sf.get("title", []),
            })
            if max_questions and len(questions) >= max_questions:
                break

        logger.info("Extracted %s questions from '%s'", len(questions), split)
        return questions

    # ------------------------------------------------------------------
    # Internal: passage extraction (works for HotpotQA context schema)
    # ------------------------------------------------------------------
    def _extract_passages(self, item: Dict) -> Sequence[Dict]:
        ctx = item.get("context") or {}
        titles = ctx.get("title") or []
        sents_per_title = ctx.get("sentences") or []
        sf = item.get("supporting_facts") or {}
        sf_titles = set(sf.get("title") or [])

        out = []
        for title, sents in zip(titles, sents_per_title):
            body = " ".join(s.strip() for s in sents if str(s).strip())
            if not body:
                continue
            out.append({
                "text": f"Title: {title}\n{body}",
                "source": "hotpot_context",
                "title": title,
                "is_supporting": title in sf_titles,
                "question_id": item.get("id", ""),
            })
        return out
