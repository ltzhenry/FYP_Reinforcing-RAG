"""Stage 2 — Multi-hop evidence retrieval."""
import logging
from typing import Dict, List

from core.embedder import Embedder
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MultiHopRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 top_k: int = 5, max_hops: int = 3,
                 similarity_threshold: float = 0.35,
                 keyword_top_k: int = 8):
        self.vs = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.max_hops = max_hops
        self.threshold = similarity_threshold
        self.keyword_top_k = keyword_top_k

    def retrieve(self, subqueries: List[Dict], extra_queries: List[str] = None) -> Dict:
        """Run multi-hop retrieval for each sub-query. *extra_queries* are
        additional retrieval probes injected by the decision engine."""
        all_evidence: List[Dict] = []
        stats = {"total_retrievals": 0, "hops_total": 0, "successful": 0}
        subquery_details = []

        queries = list(subqueries)
        if extra_queries:
            for i, eq in enumerate(extra_queries, len(queries) + 1):
                queries.append({"subquery": eq, "type": "supplement", "order": i, "dependency": None})

        for sq in queries:
            hop_result = self._multi_hop(sq["subquery"])
            subquery_details.append({
                "subquery": sq["subquery"],
                "type": sq["type"],
                "order": sq["order"],
                "hops": hop_result["hops"],
                "evidence": hop_result["evidence"],
            })
            all_evidence.extend(hop_result["evidence"])
            stats["total_retrievals"] += hop_result["retrieval_count"]
            stats["hops_total"] += hop_result["hops_performed"]
            if hop_result["evidence"]:
                stats["successful"] += 1

        unique = self._dedup(all_evidence)
        sims = [e["similarity"] for e in unique]
        stats["avg_similarity"] = sum(sims) / len(sims) if sims else 0.0
        stats["high_quality"] = sum(1 for s in sims if s >= self.threshold)

        return {
            "evidence": unique,
            "subquery_details": subquery_details,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    def _multi_hop(self, query: str) -> Dict:
        evidence: List[Dict] = []
        hops_info = []
        seen = set()
        current = query

        for hop in range(self.max_hops):
            emb = self.embedder.embed_single(current)
            results = self.vs.search(emb, self.top_k, query_text=current,
                                     keyword_top_k=self.keyword_top_k)
            filtered = [(p, s) for p, s in results if s >= self.threshold]

            new_evidence = []
            for p, sim in filtered:
                pid = p.get("id", hash(p.get("text", "")))
                if pid not in seen:
                    seen.add(pid)
                    item = {
                        "text": p["text"],
                        "similarity": sim,
                        "hop": hop + 1,
                        "source_query": current,
                        "source": p.get("source", "unknown"),
                        "title": p.get("title", ""),
                    }
                    new_evidence.append(item)
                    evidence.append(item)

            hops_info.append({"hop": hop + 1, "query": current[:120], "found": len(new_evidence)})

            high_conf = sum(1 for e in new_evidence if e["similarity"] >= 0.65)
            if high_conf >= 3:
                break

            if hop < self.max_hops - 1 and new_evidence:
                top_text = new_evidence[0]["text"]
                current = query + " " + " ".join(top_text.split()[:50])
            else:
                break

        return {
            "evidence": evidence,
            "hops": hops_info,
            "hops_performed": len(hops_info),
            "retrieval_count": len(hops_info),
        }

    @staticmethod
    def _dedup(evidence: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for e in evidence:
            h = hash(e["text"])
            if h not in seen:
                seen.add(h)
                out.append(e)
        return out
