"""Entry point: build index, evaluate, demo, or interactive mode."""
import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from config import Config
from core.data_loader import DataLoader
from core.llm_provider import get_llm_status
from evaluator import RAGEvaluator
from reasoning_rag import ReasoningRAG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-28s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================
def ensure_data(loader: DataLoader):
    if loader.train_data is None or loader.test_data is None:
        loader.load_dataset()


def build_index(rag: ReasoningRAG, loader: DataLoader,
                path: str, max_passages: int, full: bool):
    ensure_data(loader)
    if full:
        passages = loader.get_passages("train") + loader.get_passages("test")
    else:
        passages = loader.get_passages("train", max_passages=max_passages)
    if not passages:
        logger.error("No passages — cannot build index.")
        return
    rag.build_index(passages)
    rag.save_index(path)
    logger.info("Index saved → %s (%s passages)", path, len(passages))


def ensure_index(rag: ReasoningRAG, loader: DataLoader, args):
    need = args.mode == "build" or args.rebuild_index or not os.path.exists(args.index_path)
    if need:
        build_index(rag, loader, args.index_path, args.max_passages, args.full_index)
    else:
        rag.load_index(args.index_path)


def sample_questions(loader: DataLoader, n: int, seed: int):
    ensure_data(loader)
    qs = loader.get_questions("test")
    if len(qs) <= n:
        return qs
    return random.Random(seed).sample(qs, n)


# ======================================================================
# Modes
# ======================================================================
def run_eval(rag: ReasoningRAG, questions, name: str = "ReasoningRAG"):
    results = []
    for i, q in enumerate(questions, 1):
        logger.info("[%s] %s/%s — %s", name, i, len(questions), q["question"][:80])
        results.append(rag.query(q["question"]))
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_batch(results, questions, name)
    evaluator.print_summary(metrics)
    return metrics, results


def run_demo(rag: ReasoningRAG, questions):
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*70}\nDEMO {i}/{len(questions)}\nQ: {q['question']}\n{'='*70}")
        result = rag.query(q["question"])
        a = result["answer"]
        print(f"Answer  : {a['answer'][:300]}")
        print(f"Confidence: {a['confidence']:.3f}  |  Iterations: {result['metadata']['iterations']}")
        print(f"Fallback: {a['is_fallback']}  |  Method: {a['method']}")
        print(f"Evidence pieces: {len(result['aggregation']['selected'])}")
        if i < len(questions):
            input("Press Enter for next …\n")


def run_interactive(rag: ReasoningRAG):
    print("\nReasoning RAG — Interactive Mode  (type 'quit' to exit)\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        result = rag.query(q)
        a = result["answer"]
        print(f"\nAnswer ({a['method']}, conf={a['confidence']:.2f}, "
              f"iters={result['metadata']['iterations']}, "
              f"fallback={a['is_fallback']}):\n{a['answer']}\n")


# ======================================================================
# Main
# ======================================================================
def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Reasoning RAG")
    parser.add_argument("--mode", default="demo",
                        choices=["build", "eval", "demo", "interactive"])
    parser.add_argument("--index-path", default="./hotpotqa_index.pkl")
    parser.add_argument("--eval-size", type=int, default=cfg.TEST_SAMPLE_SIZE)
    parser.add_argument("--demo-size", type=int, default=cfg.DEMO_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--max-passages", type=int, default=2000)
    parser.add_argument("--full-index", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    loader = DataLoader(random_seed=args.seed)
    rag = ReasoningRAG(cfg)

    for role in ("decomposition", "generation", "judge_a", "judge_b", "decision"):
        st = get_llm_status(role)
        logger.info("LLM %s: enabled=%s model=%s", role, st["enabled"], st["model_name"])

    ensure_index(rag, loader, args)

    if args.mode == "build":
        return

    if args.mode == "eval":
        qs = sample_questions(loader, args.eval_size, args.seed)
        metrics, results = run_eval(rag, qs)
        out_dir = Path(cfg.EXPERIMENT_OUTPUT_DIR) / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        logger.info("Results saved → %s", out_dir)

    elif args.mode == "demo":
        qs = sample_questions(loader, args.demo_size, args.seed)
        run_demo(rag, qs)

    elif args.mode == "interactive":
        run_interactive(rag)


if __name__ == "__main__":
    main()
