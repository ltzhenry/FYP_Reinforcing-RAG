"""FAISS-backed vector store with hybrid (dense + keyword) search."""
import logging
import pickle
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.passages: List[Dict] = []
        self.metadata: Dict = {}

    # ---- mutate -----------------------------------------------------------
    def add_passages(self, embeddings: np.ndarray, passages: List[Dict]):
        self.index.add(embeddings.astype("float32"))
        self.passages.extend(passages)
        logger.info("Vector store: %s passages total", len(self.passages))

    # ---- search -----------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               query_text: str = "", keyword_top_k: int = 8) -> List[Tuple[Dict, float]]:
        qe = query_embedding.reshape(1, -1).astype("float32")
        dists, idxs = self.index.search(qe, top_k)

        dense = []
        for idx, dist in zip(idxs[0], dists[0]):
            if 0 <= idx < len(self.passages):
                dense.append((self.passages[idx], float(1 / (1 + dist))))

        kw = self._keyword_search(query_text, keyword_top_k) if keyword_top_k else []

        combined: Dict[int, Tuple[Dict, float]] = {}
        for p, s in dense:
            combined[p.get("id", id(p))] = (p, s)
        for p, s in kw:
            key = p.get("id", id(p))
            blended = s * 0.92
            if key in combined:
                combined[key] = (combined[key][0], max(combined[key][1], blended))
            else:
                combined[key] = (p, blended)

        merged = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return merged[:max(top_k, keyword_top_k or 0)]

    # ---- persistence ------------------------------------------------------
    def save(self, filepath: str, metadata: Dict = None):
        self.metadata = metadata or self.metadata or {}
        with open(filepath, "wb") as f:
            pickle.dump({
                "index": faiss.serialize_index(self.index),
                "passages": self.passages,
                "metadata": self.metadata,
            }, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["index"])
        self.passages = data["passages"]
        self.metadata = data.get("metadata", {})

    def clone(self) -> "VectorStore":
        vs = VectorStore(self.dimension)
        vs.index = faiss.deserialize_index(faiss.serialize_index(self.index))
        vs.passages = [p.copy() for p in self.passages]
        vs.metadata = dict(self.metadata)
        return vs

    # ---- keyword fallback -------------------------------------------------
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        if not query or not self.passages:
            return []
        tokens = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        if not tokens:
            return []
        scored = []
        for p in self.passages:
            ptokens = set(re.findall(r"[A-Za-z0-9_]+", p.get("text", "").lower()))
            overlap = len(tokens & ptokens) / len(tokens) if tokens else 0
            if overlap > 0:
                scored.append((p, min(overlap, 1.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
