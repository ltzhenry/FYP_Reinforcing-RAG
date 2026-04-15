"""LLM provider helpers — supports multiple named roles."""
import os
from typing import Dict, Optional, Tuple

from openai import OpenAI
from env_utils import load_project_env


def get_llm_client(role: str = "default") -> Tuple[Optional[OpenAI], Optional[str]]:
    """Build an OpenAI-compatible client.

    *role* is informational only — all roles share the same key/base today,
    but the signature lets us split later (e.g. judge_a vs judge_b keys).
    """
    load_project_env(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), "openai"


def get_model_name(role: str, provider: Optional[str] = "openai") -> Optional[str]:
    if provider != "openai":
        return None

    env_map = {
        "decomposition": "OPENAI_DECOMPOSITION_MODEL",
        "generation": "OPENAI_GENERATION_MODEL",
        "judge_a": "OPENAI_JUDGE_A_MODEL",
        "judge_b": "OPENAI_JUDGE_B_MODEL",
        "decision": "OPENAI_DECISION_MODEL",
    }
    env_key = env_map.get(role)
    if env_key:
        val = os.getenv(env_key)
        if val:
            return val

    return os.getenv("OPENAI_MODEL", "gpt-5.4")


def get_token_limit_kwargs(model_name: Optional[str], output_tokens: int) -> Dict[str, int]:
    if not model_name:
        return {}
    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": output_tokens}
    return {"max_tokens": output_tokens}


def get_llm_status(role: str) -> Dict[str, Optional[str]]:
    _, provider = get_llm_client(role)
    model_name = get_model_name(role, provider)
    return {
        "provider": provider,
        "model_name": model_name,
        "enabled": bool(provider and model_name),
    }
