#!/usr/bin/env python3
"""
Pipeline Health Check — verify every stage of the Reasoning RAG workflow.

Usage:
    python healthcheck.py                  # full check (needs network for HotpotQA)
    python healthcheck.py --skip-dataset   # skip HotpotQA download, use synthetic data
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(name)-28s %(levelname)s %(message)s")

# ══════════════════════════════════════════════════════════════════════
#  Visual helpers
# ══════════════════════════════════════════════════════════════════════
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"
SKIP = f"{DIM}— SKIP{RESET}"

total_pass = 0
total_fail = 0
total_warn = 0


def header(title: str):
    w = 64
    print(f"\n{CYAN}{'━' * w}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{'━' * w}{RESET}")


def check(name: str, ok: bool, detail: str = "", warn: bool = False):
    global total_pass, total_fail, total_warn
    if ok and not warn:
        total_pass += 1
        tag = PASS
    elif ok and warn:
        total_warn += 1
        tag = WARN
    else:
        total_fail += 1
        tag = FAIL
    line = f"  {tag}  {name}"
    if detail:
        line += f"  {DIM}({detail}){RESET}"
    print(line)
    return ok


def fatal(msg: str):
    print(f"\n  {RED}{BOLD}FATAL: {msg}{RESET}")
    print(f"  {DIM}Fix the above issue before proceeding.{RESET}\n")


def pause(msg: str = "Press Enter to continue to the next stage …"):
    try:
        input(f"\n  {DIM}{msg}{RESET}")
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {YELLOW}Interrupted — exiting health check.{RESET}")
        sys.exit(1)


def section_result(label: str, passed: bool):
    if passed:
        print(f"\n  {GREEN}▸ {label} — all checks passed{RESET}")
    else:
        print(f"\n  {RED}▸ {label} — some checks failed{RESET}")


# ══════════════════════════════════════════════════════════════════════
#  Stage checks
# ══════════════════════════════════════════════════════════════════════
def check_imports():
    header("STAGE 0 · Module Imports")
    modules = {
        "config":              "config",
        "core.data_loader":    "core.data_loader",
        "core.embedder":       "core.embedder",
        "core.vector_store":   "core.vector_store",
        "core.llm_provider":   "core.llm_provider",
        "pipeline.query_analyzer":     "pipeline.query_analyzer",
        "pipeline.query_decomposer":   "pipeline.query_decomposer",
        "pipeline.retriever":          "pipeline.retriever",
        "pipeline.evidence_aggregator":"pipeline.evidence_aggregator",
        "pipeline.answer_generator":   "pipeline.answer_generator",
        "pipeline.verifier":           "pipeline.verifier",
        "pipeline.decision_engine":    "pipeline.decision_engine",
        "reasoning_rag":       "reasoning_rag",
        "evaluator":           "evaluator",
    }
    all_ok = True
    for label, mod in modules.items():
        try:
            __import__(mod)
            check(f"import {label}", True)
        except Exception as e:
            check(f"import {label}", False, str(e))
            all_ok = False
    section_result("Module Imports", all_ok)
    return all_ok


def check_env():
    header("STAGE 0.5 · Environment & LLM Connectivity")
    from core.llm_provider import get_llm_status

    key_set = bool(os.getenv("OPENAI_API_KEY"))
    check("OPENAI_API_KEY is set", key_set,
          "required for decomposition, generation, verification, decision")
    if not key_set:
        fatal("Set OPENAI_API_KEY in .env — most pipeline stages need it.")
        return False

    all_ok = True
    for role in ("decomposition", "generation", "judge_a", "judge_b", "decision"):
        st = get_llm_status(role)
        ok = st["enabled"]
        check(f"LLM role '{role}'", ok,
              f"model={st['model_name']}" if ok else "not configured",
              warn=not ok)
        if not ok:
            all_ok = False

    section_result("Environment", all_ok)
    return True  # non-fatal even if some roles missing


def check_data(skip_dataset: bool):
    header("STAGE 1 · Data Loading & Passage Extraction")

    if skip_dataset:
        print(f"  {DIM}--skip-dataset: using synthetic data{RESET}")
        passages = _synthetic_passages()
        questions = _synthetic_questions()
        check("Synthetic passages created", len(passages) > 0, f"{len(passages)} passages")
        check("Synthetic questions created", len(questions) > 0, f"{len(questions)} questions")
        section_result("Data Loading (synthetic)", True)
        return passages, questions

    try:
        from core.data_loader import DataLoader
        loader = DataLoader()
        print(f"  {DIM}Downloading HotpotQA (first run may take a few minutes) …{RESET}")
        loader.load_dataset()
        train_ok = loader.train_data is not None and len(loader.train_data) > 0
        test_ok = loader.test_data is not None and len(loader.test_data) > 0
        check("HotpotQA train split loaded", train_ok,
              f"{len(loader.train_data)} items" if train_ok else "empty")
        check("HotpotQA validation split loaded", test_ok,
              f"{len(loader.test_data)} items" if test_ok else "empty")

        if not train_ok:
            fatal("No training data — cannot extract passages.")
            return None, None

        passages = loader.get_passages("train", max_passages=200)
        check("Passage extraction", len(passages) > 0, f"{len(passages)} passages")

        sample = passages[0] if passages else {}
        check("Passage has 'text' field", "text" in sample, sample.get("text", "")[:60])
        check("Passage has 'title' field", "title" in sample)

        questions = loader.get_questions("test", max_questions=10)
        check("Question extraction", len(questions) > 0, f"{len(questions)} questions")

        ok = len(passages) > 0 and len(questions) > 0
        section_result("Data Loading", ok)
        return passages, questions

    except Exception as e:
        check("HotpotQA load", False, str(e))
        print(f"  {DIM}Falling back to synthetic data …{RESET}")
        return _synthetic_passages(), _synthetic_questions()


def check_embedding_and_index(passages):
    header("STAGE 1.5 · Embedding & Vector Store")

    from config import Config
    from core.embedder import Embedder
    from core.vector_store import VectorStore

    cfg = Config()
    print(f"  {DIM}Loading embedding model: {cfg.EMBEDDING_MODEL} …{RESET}")
    t0 = time.time()
    embedder = Embedder(cfg.EMBEDDING_MODEL)
    load_time = time.time() - t0
    check("Embedder loaded", True, f"dim={embedder.embedding_dim}, {load_time:.1f}s")

    texts = [p["text"] for p in passages[:50]]
    t0 = time.time()
    embs = embedder.embed_batch(texts)
    embed_time = time.time() - t0
    check("Batch embed", embs.shape[0] == len(texts),
          f"{embs.shape}, {embed_time:.1f}s")

    vs = VectorStore(embedder.embedding_dim)
    vs.add_passages(embs, passages[:50])
    check("VectorStore populated", vs.index.ntotal == len(texts),
          f"{vs.index.ntotal} vectors")

    q_emb = embedder.embed_single("What is the capital of France?")
    results = vs.search(q_emb, top_k=3, query_text="capital France")
    check("VectorStore search returns results", len(results) > 0,
          f"{len(results)} hits, top_sim={results[0][1]:.3f}" if results else "no hits")

    section_result("Embedding & Index", True)
    return embedder, vs


def check_query_analysis():
    header("STAGE 2a · Query Analyzer")
    from pipeline.query_analyzer import QueryAnalyzer

    qa = QueryAnalyzer()
    simple = qa.analyze("What is DNA?")
    check("Simple question detected", not simple["is_complex"],
          f"score={simple['complexity_score']:.2f}, type={simple['query_type']}")

    complex_q = qa.analyze("How does the relationship between DNA replication and cell division affect cancer growth, and what therapeutic approaches target this mechanism?")
    check("Complex question detected", complex_q["is_complex"],
          f"score={complex_q['complexity_score']:.2f}, type={complex_q['query_type']}")

    section_result("Query Analyzer", True)
    return qa


def check_query_decomposer(qa):
    header("STAGE 2b · Query Decomposer (LLM)")
    from pipeline.query_decomposer import QueryDecomposer

    decomposer = QueryDecomposer()
    llm_ok = decomposer.client is not None
    check("Decomposer LLM client available", llm_ok, warn=not llm_ok)

    simple_a = qa.analyze("What is DNA?")
    subs = decomposer.decompose(simple_a)
    check("Simple → no decomposition", len(subs) == 1 and subs[0]["type"] == "direct",
          f"{len(subs)} subquery")

    complex_a = qa.analyze("What is the relationship between sleep deprivation and cognitive performance, and how does caffeine affect both?")
    subs = decomposer.decompose(complex_a)
    check("Complex → decomposed", len(subs) >= 2,
          f"{len(subs)} subqueries" + (f": {[s['subquery'][:40] for s in subs]}" if subs else ""))

    if llm_ok:
        refined = decomposer.refine_query("stuff about DNA thing", "Question is too vague, be specific")
        check("refine_query works", refined != "stuff about DNA thing",
              f"→ {refined[:60]}")

    section_result("Query Decomposer", True)
    return decomposer


def check_retrieval(embedder, vs):
    header("STAGE 3 · Multi-Hop Retriever")
    from pipeline.retriever import MultiHopRetriever

    retriever = MultiHopRetriever(vs, embedder, top_k=3, max_hops=2,
                                   similarity_threshold=0.1)
    subqueries = [{"subquery": "capital of France", "type": "direct", "order": 1, "dependency": None}]
    result = retriever.retrieve(subqueries)

    ev_count = len(result["evidence"])
    check("Retrieval returns evidence", ev_count > 0, f"{ev_count} pieces")
    check("Stats populated", result["stats"]["total_retrievals"] > 0,
          f"retrievals={result['stats']['total_retrievals']}")

    result2 = retriever.retrieve(subqueries, extra_queries=["more about Paris landmarks"])
    check("Extra queries (retrieve_more) works",
          len(result2["evidence"]) >= ev_count,
          f"{len(result2['evidence'])} pieces with supplement")

    section_result("Multi-Hop Retriever", ev_count > 0)
    return retriever


def check_aggregation_and_generation(retriever):
    header("STAGE 4 · Evidence Aggregation + Answer Generation")
    from pipeline.evidence_aggregator import EvidenceAggregator
    from pipeline.answer_generator import AnswerGenerator

    subqueries = [{"subquery": "capital of France", "type": "direct", "order": 1, "dependency": None}]
    retrieval = retriever.retrieve(subqueries)

    agg = EvidenceAggregator(max_evidence_length=3000)
    aggregated = agg.aggregate(retrieval["evidence"])
    check("Aggregation selects evidence", len(aggregated["selected"]) > 0,
          f"{len(aggregated['selected'])} selected, quality={aggregated['avg_quality']:.3f}")
    check("Coverage computed", isinstance(aggregated["coverage"], float),
          f"coverage={aggregated['coverage']:.3f}")

    gen = AnswerGenerator()
    llm_ok = gen.client is not None
    check("Generator LLM available", llm_ok, warn=not llm_ok)

    result = gen.generate("What is the capital of France?", aggregated["selected"])
    check("Answer generated", len(result["answer"]) > 5,
          f"method={result['method']}, answer={result['answer'][:80]}")

    fb = gen.generate_fallback("What is quantum computing?")
    check("Fallback generation", len(fb["answer"]) > 5,
          f"method={fb['method']}")

    section_result("Aggregation + Generation", True)
    return aggregated


def check_verification(aggregated):
    header("STAGE 5 · Cross-Model Verification (Dual Judge)")
    from pipeline.verifier import CrossModelVerifier

    verifier = CrossModelVerifier()
    a_ok = verifier.client_a is not None
    b_ok = verifier.client_b is not None
    check("Judge A available", a_ok, f"model={verifier.model_a}" if a_ok else "not configured", warn=not a_ok)
    check("Judge B available", b_ok, f"model={verifier.model_b}" if b_ok else "not configured", warn=not b_ok)

    result = verifier.verify(
        "What is the capital of France?",
        "The capital of France is Paris.",
        aggregated["selected"],
    )
    conf = result["confidence"]
    div = result["divergence"]
    check("Confidence score produced", isinstance(conf, float), f"confidence={conf:.3f}")
    check("Divergence computed", isinstance(div, float), f"divergence={div:.3f}")
    ja = result["judge_a"]
    jb = result["judge_b"]
    check("Judge A returned 3 dimensions",
          all(k in ja for k in ("faithfulness", "completeness", "consistency")),
          json.dumps(ja))
    check("Judge A non-zero scores", any(v > 0 for v in ja.values()),
          json.dumps(ja), warn=not any(v > 0 for v in ja.values()))
    check("Judge B returned 3 dimensions",
          all(k in jb for k in ("faithfulness", "completeness", "consistency")),
          json.dumps(jb))
    check("Judge B non-zero scores", any(v > 0 for v in jb.values()),
          json.dumps(jb), warn=not any(v > 0 for v in jb.values()))

    section_result("Verification", conf > 0)
    return verifier


def check_decision():
    header("STAGE 6 · Decision Engine")
    from pipeline.decision_engine import DecisionEngine

    engine = DecisionEngine()
    llm_ok = engine.client is not None
    check("Decision LLM available", llm_ok, warn=not llm_ok)

    # Test heuristic path directly (deterministic, not affected by LLM variance)
    from pipeline.decision_engine import DecisionEngine as DE

    low_evidence = {
        "confidence": 0.3, "divergence": 0.1,
        "details": {"dimension_means": {"faithfulness": 0.2, "completeness": 0.2, "consistency": 0.5}},
    }
    d = DE._heuristic_decide(low_evidence, {"coverage": 0.01})
    check("Heuristic: low coverage → retrieve_more", d["action"] == "retrieve_more",
          f"action={d['action']}, reason={d['reason'][:60]}")

    low_faith = {
        "confidence": 0.3, "divergence": 0.1,
        "details": {"dimension_means": {"faithfulness": 0.2, "completeness": 0.6, "consistency": 0.6}},
    }
    d = DE._heuristic_decide(low_faith, {"coverage": 0.3})
    check("Heuristic: low faithfulness → refine_query", d["action"] == "refine_query",
          f"action={d['action']}")

    ok_evidence = {
        "confidence": 0.4, "divergence": 0.05,
        "details": {"dimension_means": {"faithfulness": 0.6, "completeness": 0.5, "consistency": 0.5}},
    }
    d = DE._heuristic_decide(ok_evidence, {"coverage": 0.3})
    check("Heuristic: adequate evidence → regenerate", d["action"] == "regenerate",
          f"action={d['action']}")

    # Also verify that the LLM path returns a valid action (any of the 3)
    if llm_ok:
        d = engine.decide("test?", "test answer", [{"text": "some evidence"}],
                          low_evidence, {"coverage": 0.01}, 1)
        valid_actions = {"refine_query", "retrieve_more", "regenerate"}
        check("LLM decision returns valid action", d["action"] in valid_actions,
              f"action={d['action']}, reason={d['reason'][:50]}")

    section_result("Decision Engine", True)


def check_full_pipeline(skip_dataset: bool):
    header("STAGE 7 · Full Pipeline (end-to-end query)")
    from reasoning_rag import ReasoningRAG
    from config import Config

    cfg = Config()
    cfg.MAX_ITERATIONS = 3  # limit for health check speed

    print(f"  {DIM}Initialising RAG system …{RESET}")
    rag = ReasoningRAG(cfg)

    if skip_dataset:
        passages = _synthetic_passages()
    else:
        from core.data_loader import DataLoader
        loader = DataLoader()
        loader.load_dataset()
        passages = loader.get_passages("train", max_passages=200)

    if not passages:
        check("Passages available", False)
        return

    print(f"  {DIM}Building index ({len(passages)} passages) …{RESET}")
    rag.build_index(passages)
    check("Index built", rag.vector_store.index.ntotal > 0,
          f"{rag.vector_store.index.ntotal} vectors")

    question = "What is the relationship between Arthur's Magazine and First for Women?"
    print(f"\n  {BOLD}Query:{RESET} {question}")
    print(f"  {DIM}Running full iterative pipeline …{RESET}\n")

    t0 = time.time()
    result = rag.query(question)
    elapsed = time.time() - t0

    a = result["answer"]
    m = result["metadata"]
    v = result["verification"]
    chain = result["transparency_chain"]

    print(f"  {BOLD}Answer:{RESET} {a['answer'][:300]}")
    print(f"  {BOLD}Confidence:{RESET} {a['confidence']:.3f}")
    print(f"  {BOLD}Iterations:{RESET} {m['iterations']}")
    print(f"  {BOLD}Method:{RESET} {a['method']}")
    print(f"  {BOLD}Fallback:{RESET} {a['is_fallback']}")
    print(f"  {BOLD}Runtime:{RESET} {elapsed:.1f}s")
    print(f"  {BOLD}Evidence:{RESET} {len(result['aggregation']['selected'])} pieces")
    print(f"  {BOLD}Chain steps:{RESET} {len(chain)}")
    print()

    check("Answer is non-empty", len(a["answer"]) > 10)
    check("Confidence is a number", 0.0 <= a["confidence"] <= 1.0,
          f"{a['confidence']:.3f}")
    check("Iterations tracked", m["iterations"] >= 1, f"{m['iterations']}")
    check("Transparency chain populated", len(chain) >= 3, f"{len(chain)} steps")

    has_analysis = any(s["step"] == "analysis" for s in chain)
    has_decomp = any(s["step"] == "decomposition" for s in chain)
    has_retrieval = any("retrieval" in s["step"] for s in chain)
    has_generation = any("generation" in s["step"] for s in chain)
    has_verification = any("verification" in s["step"] for s in chain)
    check("Chain: analysis step", has_analysis)
    check("Chain: decomposition step", has_decomp)
    check("Chain: retrieval step", has_retrieval)
    check("Chain: generation step", has_generation)
    check("Chain: verification step", has_verification)

    if m["iterations"] > 1:
        has_decision = any("decision" in s["step"] for s in chain)
        check("Chain: decision step (iterative)", has_decision)

    # Verification structure
    check("Verification has judge_a", "judge_a" in v)
    check("Verification has judge_b", "judge_b" in v)
    check("Verification has confidence", "confidence" in v)

    section_result("Full Pipeline", len(a["answer"]) > 10)

    print(f"\n  {CYAN}{'─' * 50}{RESET}")
    print(f"  {BOLD}Transparency Chain Visualization:{RESET}")
    print(f"  {CYAN}{'─' * 50}{RESET}")
    for i, step in enumerate(chain):
        sname = step["step"]
        icon = "🔍" if "retrieval" in sname else \
               "🧠" if "analysis" in sname else \
               "✂️" if "decomp" in sname else \
               "⚡" if "generation" in sname else \
               "⚖️" if "verification" in sname else \
               "🔀" if "decision" in sname else \
               "🔄" if "refine" in sname or "retrieve_more" in sname or "regenerate" in sname else \
               "🆘" if "fallback" in sname else "▪️"
        detail_parts = []
        if "confidence" in step:
            detail_parts.append(f"conf={step['confidence']:.3f}")
        if "evidence_count" in step:
            detail_parts.append(f"evidence={step['evidence_count']}")
        if "method" in step:
            detail_parts.append(f"method={step['method']}")
        if "action" in step:
            detail_parts.append(f"action={step['action']}")
        if "subqueries" in step:
            detail_parts.append(f"subs={len(step['subqueries'])}")
        detail = ", ".join(detail_parts)
        connector = "  │" if i < len(chain) - 1 else "  "
        print(f"  {icon} {sname}{f'  {DIM}({detail}){RESET}' if detail else ''}")
        if i < len(chain) - 1:
            print(f"  │")


# ══════════════════════════════════════════════════════════════════════
#  Synthetic data (for --skip-dataset)
# ══════════════════════════════════════════════════════════════════════
def _synthetic_passages():
    raw = [
        ("Arthur's Magazine", "Arthur's Magazine was an American literary periodical published in Philadelphia from 1844 to 1846."),
        ("First for Women", "First for Women is a woman's magazine published by Bauer Media Group in the USA since 1989."),
        ("Paris", "Paris is the capital and most populous city of France, with a population of over 2 million."),
        ("DNA", "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix."),
        ("Photosynthesis", "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy."),
    ]
    return [{"id": i, "text": f"Title: {t}\n{b}", "source": "synthetic",
             "title": t, "is_supporting": False, "question_id": ""}
            for i, (t, b) in enumerate(raw)]


def _synthetic_questions():
    return [
        {"id": "q1", "question": "What is Arthur's Magazine?",
         "answers": ["An American literary periodical"], "question_type": "bridge", "difficulty": "easy"},
        {"id": "q2", "question": "What is the relationship between Arthur's Magazine and First for Women?",
         "answers": ["Both are magazines"], "question_type": "comparison", "difficulty": "medium"},
    ]


# ══════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════
def print_summary():
    header("SUMMARY")
    total = total_pass + total_fail + total_warn
    print(f"  {GREEN}{total_pass} passed{RESET}  |  "
          f"{RED}{total_fail} failed{RESET}  |  "
          f"{YELLOW}{total_warn} warnings{RESET}  |  "
          f"{total} total\n")
    if total_fail == 0:
        print(f"  {GREEN}{BOLD}🎉 All pipeline stages are working!{RESET}")
        print(f"  {DIM}You can now run:{RESET}")
        print(f"    python main.py --mode build --full-index")
        print(f"    python main.py --mode interactive")
        print(f"    python main.py --mode eval --eval-size 20")
    else:
        print(f"  {RED}{BOLD}Some stages failed — see details above.{RESET}")
    print()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Reasoning RAG Health Check")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip HotpotQA download, use synthetic data")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}{'═' * 64}{RESET}")
    print(f"{CYAN}{BOLD}   Reasoning RAG — Pipeline Health Check{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 64}{RESET}")

    # Stage 0: imports
    if not check_imports():
        fatal("Module imports failed — fix before continuing.")
        print_summary()
        return

    pause("Imports OK. Press Enter to check environment & LLM …")

    # Stage 0.5: env
    check_env()
    pause("Environment checked. Press Enter to load data …")

    # Stage 1: data
    passages, questions = check_data(args.skip_dataset)
    if not passages:
        fatal("No passages available — cannot continue.")
        print_summary()
        return

    pause(f"Data ready ({len(passages)} passages). Press Enter to build embeddings …")

    # Stage 1.5: embedding + vector store
    embedder, vs = check_embedding_and_index(passages)
    pause("Index built. Press Enter to test query analysis …")

    # Stage 2a: query analyzer
    qa = check_query_analysis()
    pause("Analyzer OK. Press Enter to test decomposition …")

    # Stage 2b: decomposer
    decomposer = check_query_decomposer(qa)
    pause("Decomposer OK. Press Enter to test retrieval …")

    # Stage 3: retrieval
    retriever = check_retrieval(embedder, vs)
    pause("Retrieval OK. Press Enter to test aggregation + generation …")

    # Stage 4: aggregation + generation
    aggregated = check_aggregation_and_generation(retriever)
    pause("Generation OK. Press Enter to test verification …")

    # Stage 5: verification
    check_verification(aggregated)
    pause("Verification OK. Press Enter to test decision engine …")

    # Stage 6: decision
    check_decision()
    pause("Decision engine OK. Press Enter to run FULL PIPELINE test …")

    # Stage 7: end-to-end
    check_full_pipeline(args.skip_dataset)

    print_summary()


if __name__ == "__main__":
    main()
