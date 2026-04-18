"""FAISS dense + BM25 sparse hybrid retrieval with Reciprocal Rank Fusion."""
import logging
import pickle
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.passages: List[Dict] = []
        self.metadata: Dict = {}
        self.bm25: BM25Okapi = None
        self._tokenized_corpus: List[List[str]] = []

    # ---- mutate -----------------------------------------------------------
    def add_passages(self, embeddings: np.ndarray, passages: List[Dict]):
        self.index.add(embeddings.astype("float32"))
        self.passages.extend(passages)
        self._rebuild_bm25()
        logger.info("Vector store: %s passages total (BM25 re-indexed)", len(self.passages))

    def _rebuild_bm25(self):
        self._tokenized_corpus = [_tokenize(p.get("text", "")) for p in self.passages]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    # ---- search -----------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               query_text: str = "", keyword_top_k: int = 8) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval: dense (FAISS) + sparse (BM25), fused with RRF."""
        # --- dense branch ---
        dense_k = max(top_k * 2, 20)
        qe = query_embedding.reshape(1, -1).astype("float32")
        dists, idxs = self.index.search(qe, dense_k)
        dense_ranking = []
        for idx, dist in zip(idxs[0], dists[0]):
            if 0 <= idx < len(self.passages):
                dense_ranking.append((idx, float(1 / (1 + dist))))

        # --- sparse branch (BM25) ---
        sparse_ranking = []
        if self.bm25 and query_text and keyword_top_k:
            query_tokens = _tokenize(query_text)
            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)
                sparse_k = max(keyword_top_k * 2, 20)
                top_sparse_idxs = np.argsort(scores)[::-1][:sparse_k]
                max_score = scores[top_sparse_idxs[0]] if len(top_sparse_idxs) else 1.0
                for idx in top_sparse_idxs:
                    if scores[idx] > 0:
                        normalized = float(scores[idx] / max_score) if max_score > 0 else 0
                        sparse_ranking.append((int(idx), normalized))

        # --- Reciprocal Rank Fusion ---
        return self._rrf_fuse(dense_ranking, sparse_ranking, top_k)

    def _rrf_fuse(self, dense_ranking: List[Tuple[int, float]],
                  sparse_ranking: List[Tuple[int, float]],
                  top_k: int, k_constant: int = 60) -> List[Tuple[Dict, float]]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank_i)) across rankers.

        Also keeps max raw score per passage for downstream similarity use.
        """
        rrf_scores: Dict[int, float] = {}
        raw_scores: Dict[int, float] = {}

        for rank, (pid, score) in enumerate(dense_ranking, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_constant + rank)
            raw_scores[pid] = max(raw_scores.get(pid, 0.0), score)

        for rank, (pid, score) in enumerate(sparse_ranking, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_constant + rank)
            raw_scores[pid] = max(raw_scores.get(pid, 0.0), score)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Tuple[Dict, float]] = []
        for pid, _rrf in ranked[:top_k]:
            if 0 <= pid < len(self.passages):
                results.append((self.passages[pid], raw_scores[pid]))
        return results

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
        self._rebuild_bm25()

    def clone(self) -> "VectorStore":
        vs = VectorStore(self.dimension)
        vs.index = faiss.deserialize_index(faiss.serialize_index(self.index))
        vs.passages = [p.copy() for p in self.passages]
        vs.metadata = dict(self.metadata)
        vs._rebuild_bm25()
        return vs
