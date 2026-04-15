#!/usr/bin/env python3
"""
Benchmark: Naive RAG vs Reasoning RAG on the same HotpotQA questions.

Usage:
    python benchmark.py                           # 20 questions, default seed
    python benchmark.py --eval-size 50 --seed 42  # 50 questions
    python benchmark.py --skip-dataset             # use synthetic data (fast, no download)
"""
import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-28s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
#  Terminal colours
# ══════════════════════════════════════════════════════════════════════
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m"; D = "\033[2m"; RST = "\033[0m"


def bar(value: float, width: int = 30, label: str = "") -> str:
    filled = int(value * width)
    colour = G if value >= 0.5 else Y if value >= 0.25 else R
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{RST} {value:.3f}  {label}"


def delta_str(a: float, b: float) -> str:
    d = a - b
    if abs(d) < 0.001:
        return f"{D}  ={RST}"
    sign = "+" if d > 0 else ""
    colour = G if d > 0 else R
    return f"{colour}{sign}{d:.3f}{RST}"


# ══════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════
def load_questions(args):
    if args.skip_dataset:
        return _synthetic_questions()

    from core.data_loader import DataLoader
    loader = DataLoader(random_seed=args.seed)
    loader.load_dataset()
    qs = loader.get_questions("test")
    if len(qs) > args.eval_size:
        qs = random.Random(args.seed).sample(qs, args.eval_size)
    return qs


def load_passages(args, questions=None):
    if args.skip_dataset:
        return _synthetic_passages()

    from core.data_loader import DataLoader
    loader = DataLoader(random_seed=args.seed)
    loader.load_dataset()

    train_passages = loader.get_passages("train", max_passages=args.max_passages)
    eval_passages = loader.get_passages("test")

    seen = {p["text"] for p in train_passages}
    for p in eval_passages:
        if p["text"] not in seen:
            seen.add(p["text"])
            p["id"] = len(train_passages) + len(seen)
            train_passages.append(p)

    logger.info("Index: %s train + %s eval context = %s total passages",
                args.max_passages, len(eval_passages), len(train_passages))
    return train_passages


def _synthetic_passages():
    raw = [
        ("Arthur's Magazine", "Arthur's Magazine was an American literary periodical published in Philadelphia from 1844 to 1846."),
        ("First for Women", "First for Women is a woman's magazine published by Bauer Media Group in the USA since 1989."),
        ("Paris", "Paris is the capital and most populous city of France, with a population of over 2 million."),
        ("DNA", "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix."),
        ("Photosynthesis", "Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose."),
        ("Machine Learning", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience."),
        ("Quantum Computing", "Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement to perform computation."),
        ("Climate Change", "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities since the 1800s."),
    ]
    return [{"id": i, "text": f"Title: {t}\n{b}", "source": "synthetic",
             "title": t, "is_supporting": False, "question_id": ""}
            for i, (t, b) in enumerate(raw)]


def _synthetic_questions():
    return [
        {"id": "q1", "question": "What is Arthur's Magazine?",
         "answers": ["An American literary periodical published in Philadelphia"], "question_type": "bridge", "difficulty": "easy"},
        {"id": "q2", "question": "What is the relationship between Arthur's Magazine and First for Women?",
         "answers": ["Both are American magazines but from different eras"], "question_type": "comparison", "difficulty": "medium"},
        {"id": "q3", "question": "What is the capital of France?",
         "answers": ["Paris"], "question_type": "bridge", "difficulty": "easy"},
        {"id": "q4", "question": "How does photosynthesis relate to climate change?",
         "answers": ["Photosynthesis absorbs CO2 which is a greenhouse gas"], "question_type": "bridge", "difficulty": "hard"},
        {"id": "q5", "question": "What is the difference between machine learning and quantum computing?",
         "answers": ["Machine learning is about learning from data while quantum computing uses quantum mechanics for computation"], "question_type": "comparison", "difficulty": "hard"},
    ]


# ══════════════════════════════════════════════════════════════════════
#  Run evaluation for one system
# ══════════════════════════════════════════════════════════════════════
def run_system(system, questions, name: str):
    print(f"\n{C}{'━' * 64}{RST}")
    print(f"{C}{B}  Running: {name}  ({len(questions)} questions){RST}")
    print(f"{C}{'━' * 64}{RST}")

    from evaluator import RAGEvaluator

    results = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        print(f"  {D}[{name}] {i}/{len(questions)} — {q['question'][:65]}…{RST}", end="", flush=True)
        result = system.query(q["question"])
        elapsed = time.time() - t0
        conf = result["answer"]["confidence"]
        iters = result["metadata"]["iterations"]
        fb = result["metadata"]["is_fallback"]
        status = f"{G}✓{RST}" if not fb else f"{Y}⚠fb{RST}"
        print(f"  {status} {elapsed:.1f}s  conf={conf:.2f}  iters={iters}")
        results.append(result)

    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_batch(results, questions, name)
    return metrics, results


# ══════════════════════════════════════════════════════════════════════
#  Comparison visualization
# ══════════════════════════════════════════════════════════════════════
def print_comparison(m_naive: dict, m_reasoning: dict):
    print(f"\n{C}{'━' * 72}{RST}")
    print(f"{C}{B}  BENCHMARK COMPARISON: Naive RAG vs Reasoning RAG{RST}")
    print(f"{C}{'━' * 72}{RST}\n")

    metrics_to_show = [
        ("Exact Match",       "exact_match",       True),
        ("Token F1",          "token_f1",          True),
        ("Answer Coverage",   "answer_coverage",   True),
        ("Avg Confidence",    "avg_confidence",     True),
        ("Avg Iterations",    "avg_iterations",     False),
        ("Avg Latency (s)",   "avg_latency_seconds", False),
        ("Fallback Rate",     "fallback_rate",      False),
    ]

    # Header
    print(f"  {'Metric':<22} {'Naive RAG':>12} {'Reasoning RAG':>14} {'Delta':>10}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 14} {'─' * 10}")

    for label, key, higher_better in metrics_to_show:
        nv = m_naive.get(key, 0)
        rv = m_reasoning.get(key, 0)
        ds = delta_str(rv, nv) if higher_better else delta_str(nv, rv)
        print(f"  {label:<22} {nv:>12.4f} {rv:>14.4f} {ds:>10}")

    # Visual bars
    print(f"\n{C}  ── Visual Comparison ──{RST}\n")

    bar_metrics = [
        ("Exact Match",     "exact_match"),
        ("Token F1",        "token_f1"),
        ("Answer Coverage", "answer_coverage"),
        ("Avg Confidence",  "avg_confidence"),
    ]
    for label, key in bar_metrics:
        nv = m_naive.get(key, 0)
        rv = m_reasoning.get(key, 0)
        print(f"  {label}")
        print(f"    Naive     {bar(nv)}")
        print(f"    Reasoning {bar(rv)}")
        print()

    # Per-question latency comparison
    print(f"{C}  ── Efficiency ──{RST}\n")
    nv_lat = m_naive.get("avg_latency_seconds", 0)
    rv_lat = m_reasoning.get("avg_latency_seconds", 0)
    speedup = nv_lat / rv_lat if rv_lat > 0 else 0
    print(f"  Naive avg latency:     {nv_lat:.2f}s")
    print(f"  Reasoning avg latency: {rv_lat:.2f}s")
    if speedup > 1:
        print(f"  Naive is {speedup:.1f}x faster (expected — no verification loop)")
    else:
        ratio = rv_lat / nv_lat if nv_lat > 0 else 0
        print(f"  Reasoning is {ratio:.1f}x slower (expected — iterative verification)")

    print(f"\n  Reasoning avg iterations: {m_reasoning.get('avg_iterations', 0):.1f}")
    print(f"  Reasoning fallback rate:  {m_reasoning.get('fallback_rate', 0):.1%}")


def print_winner(m_naive: dict, m_reasoning: dict):
    print(f"\n{C}{'━' * 72}{RST}")
    score_n = 0
    score_r = 0
    for key in ("exact_match", "token_f1", "answer_coverage"):
        if m_reasoning.get(key, 0) > m_naive.get(key, 0) + 0.001:
            score_r += 1
        elif m_naive.get(key, 0) > m_reasoning.get(key, 0) + 0.001:
            score_n += 1

    if score_r > score_n:
        print(f"  {G}{B}🏆 Reasoning RAG wins on {score_r}/3 quality metrics{RST}")
    elif score_n > score_r:
        print(f"  {Y}{B}⚠ Naive RAG wins on {score_n}/3 quality metrics — check pipeline config{RST}")
    else:
        print(f"  {Y}{B}≈ Tie on quality metrics{RST}")
    print(f"{C}{'━' * 72}{RST}\n")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    from config import Config
    from naive_rag import NaiveRAG
    from reasoning_rag import ReasoningRAG

    cfg = Config()
    parser = argparse.ArgumentParser(description="Benchmark: Naive vs Reasoning RAG")
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--max-passages", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max iterations for Reasoning RAG (lower = faster benchmark)")
    parser.add_argument("--index-path", default="./hotpotqa_index.pkl")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    cfg.MAX_ITERATIONS = args.max_iterations

    print(f"\n{C}{B}{'═' * 72}{RST}")
    print(f"{C}{B}   Benchmark: Naive RAG vs Reasoning RAG{RST}")
    print(f"{C}{B}   eval_size={args.eval_size}  seed={args.seed}  max_iters={args.max_iterations}{RST}")
    print(f"{C}{B}{'═' * 72}{RST}")

    # ---- Build shared index -------------------------------------------
    print(f"\n  {D}Loading data …{RST}")
    questions = load_questions(args)
    passages = load_passages(args, questions)
    print(f"  {G}✓{RST} {len(passages)} passages, {len(questions)} questions\n")

    print(f"  {D}Initialising systems (shared embedder + index) …{RST}")
    reasoning = ReasoningRAG(cfg)

    if not args.rebuild_index and os.path.exists(args.index_path):
        print(f"  {D}Loading existing index: {args.index_path}{RST}")
        reasoning.load_index(args.index_path)
        if reasoning.vector_store.index.ntotal < len(passages):
            print(f"  {Y}Index has {reasoning.vector_store.index.ntotal} vectors but need {len(passages)} — rebuilding{RST}")
            reasoning = ReasoningRAG(cfg)
            reasoning.build_index(passages)
            reasoning.save_index(args.index_path)
    else:
        print(f"  {D}Building index ({len(passages)} passages) …{RST}")
        reasoning.build_index(passages)
        reasoning.save_index(args.index_path)

    naive = NaiveRAG(cfg, embedder=reasoning.embedder, vector_store=reasoning.vector_store)
    print(f"  {G}✓{RST} Both systems ready (shared {reasoning.vector_store.index.ntotal} vectors)\n")

    # ---- Run both systems ---------------------------------------------
    m_naive, r_naive = run_system(naive, questions, "Naive RAG")
    m_reasoning, r_reasoning = run_system(reasoning, questions, "Reasoning RAG")

    # ---- Comparison ---------------------------------------------------
    print_comparison(m_naive, m_reasoning)
    print_winner(m_naive, m_reasoning)

    # ---- Save results -------------------------------------------------
    out_dir = Path(cfg.EXPERIMENT_OUTPUT_DIR) / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "eval_size": len(questions),
            "seed": args.seed,
            "max_passages": args.max_passages,
            "max_iterations": args.max_iterations,
            "timestamp": datetime.now().isoformat(),
        },
        "naive_rag": m_naive,
        "reasoning_rag": m_reasoning,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=str))

    detailed = {
        "naive_results": [
            {"question": q["question"], "pred": r["answer"]["answer"][:200],
             "confidence": r["answer"]["confidence"], "method": r["answer"]["method"]}
            for q, r in zip(questions, r_naive)
        ],
        "reasoning_results": [
            {"question": q["question"], "pred": r["answer"]["answer"][:200],
             "confidence": r["answer"]["confidence"], "method": r["answer"]["method"],
             "iterations": r["metadata"]["iterations"], "is_fallback": r["metadata"]["is_fallback"]}
            for q, r in zip(questions, r_reasoning)
        ],
    }
    (out_dir / "detailed_results.json").write_text(json.dumps(detailed, indent=2, default=str))

    print(f"  {G}Results saved → {out_dir}{RST}\n")


if __name__ == "__main__":
    main()
