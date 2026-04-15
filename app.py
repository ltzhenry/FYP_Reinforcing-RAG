#!/usr/bin/env python3
"""Reasoning RAG — Web Chat Interface (Flask).

Run:   python app.py
Open:  http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template
import logging
import traceback
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(name)-28s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./hotpotqa_index.pkl")

_rag = None


def _build_rag():
    from reasoning_rag import ReasoningRAG
    from config import Config
    rag = ReasoningRAG(Config())
    if os.path.exists(INDEX_PATH):
        logger.info("Loading index from %s …", INDEX_PATH)
        rag.load_index(INDEX_PATH)
        logger.info("Index loaded (%s vectors)", rag.vector_store.index.ntotal)
    else:
        logger.error("Index NOT found: %s — run: python main.py --mode build --full-index", INDEX_PATH)
    return rag


def get_rag():
    global _rag
    if _rag is None:
        _rag = _build_rag()
    return _rag


def _safe(obj, *keys, default=None):
    for k in keys:
        if obj is None:
            return default
        try:
            obj = obj[k] if isinstance(obj, dict) else getattr(obj, k)
        except (KeyError, AttributeError, TypeError):
            return default
    return default if obj is None else obj


def to_json(result: dict, question: str) -> dict:
    answer = _safe(result, "answer", default={})
    analysis = _safe(result, "analysis", default={})
    agg = _safe(result, "aggregation", default={})
    verif = _safe(result, "verification", default={})
    meta = _safe(result, "metadata", default={})
    chain = _safe(result, "transparency_chain", default=[])
    evidence = _safe(agg, "selected", default=[])
    stats = _safe(result, "retrieval", "stats", default={})

    return {
        "question": question,
        "answer": _safe(answer, "answer", default=""),
        "confidence": float(_safe(answer, "confidence", default=0)),
        "is_fallback": bool(_safe(answer, "is_fallback", default=False)),
        "method": _safe(answer, "method", default=""),
        "iterations": _safe(meta, "iterations", default=1),
        "runtime": round(float(_safe(meta, "runtime_seconds", default=0)), 3),
        "analysis": {
            "complexity_score": float(_safe(analysis, "complexity_score", default=0)),
            "is_complex": bool(_safe(analysis, "is_complex", default=False)),
            "query_type": _safe(analysis, "query_type", default=""),
        },
        "verification": {
            "confidence": float(_safe(verif, "confidence", default=0)),
            "divergence": float(_safe(verif, "divergence", default=0)),
            "judge_a": _safe(verif, "judge_a", default={}),
            "judge_b": _safe(verif, "judge_b", default={}),
        },
        "evidence": [
            {
                "rank": i + 1,
                "quality": round(float(_safe(e, "quality_score", default=0)), 3),
                "similarity": round(float(_safe(e, "similarity", default=0)), 3),
                "hop": _safe(e, "hop", default=1),
                "source": _safe(e, "source", default=""),
                "text": _safe(e, "text", default=""),
            }
            for i, e in enumerate(evidence[:6])
        ],
        "transparency_chain": chain,
        "stats": {
            "total_retrievals": _safe(stats, "total_retrievals", default=0),
            "avg_similarity": round(float(_safe(stats, "avg_similarity", default=0)), 3),
            "evidence_count": len(evidence),
            "coverage": round(float(_safe(agg, "coverage", default=0)), 3),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    try:
        body = request.get_json(force=True) or {}
        question = body.get("question", "").strip()
        if not question:
            return jsonify({"ok": False, "error": "Question cannot be empty."}), 400

        rag = get_rag()
        if rag.vector_store.index.ntotal == 0:
            return jsonify({"ok": False, "error": "Index is empty. Build it first."}), 503

        result = rag.query(question)
        return jsonify({"ok": True, "data": to_json(result, question)})

    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": traceback.format_exc().splitlines()[-1]}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nReasoning RAG · Web Interface → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
