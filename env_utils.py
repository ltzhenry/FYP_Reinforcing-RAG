"""Locate and load .env files from the project tree."""
from pathlib import Path
import os

_loaded = False


def load_project_env(override: bool = False):
    global _loaded
    if _loaded and not override:
        return

    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for candidate in candidates:
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("\"'")
                if override or key not in os.environ:
                    os.environ[key] = value
            break

    _loaded = True
