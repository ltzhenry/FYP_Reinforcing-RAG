"""Global configuration."""
import os
from env_utils import load_project_env

load_project_env(override=True)


class Config:
    # --- Embedding ---
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    RANDOM_SEED = 42

    # --- Retrieval ---
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.35
    KEYWORD_FALLBACK_TOP_K = 8
    MAX_HOPS = 3

    # --- Decomposition ---
    MAX_SUBQUERIES = 4
    COMPLEXITY_THRESHOLD = 0.2

    # --- Evidence & generation ---
    MAX_EVIDENCE_LENGTH = 3000

    # --- Verification ---
    CONFIDENCE_THRESHOLD = 0.5
    MAX_ITERATIONS = 10

    # --- Evaluation ---
    TEST_SAMPLE_SIZE = 50
    DEMO_SAMPLE_SIZE = 3
    EXPERIMENT_OUTPUT_DIR = "./experiment_outputs"
