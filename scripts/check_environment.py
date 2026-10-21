"""Check scaffold prerequisites without generating text or modifying Ollama."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip the Ollama connection check")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        print("FAIL: Python 3.11 or newer is required.", file=sys.stderr)
        return 1

    # Allow running the checker before editable-package installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from freezer_rag.config import Settings

        settings = Settings()
    except ImportError as exc:
        print(f"FAIL: Missing dependency: {exc}. Install requirements-dev.txt.", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"FAIL: Invalid RAG configuration: {exc}", file=sys.stderr)
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"Generation model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Vector store: Qdrant local at {settings.qdrant_path}")

    failed = False
    for name in (
        "business_requirements.pdf",
        "Software_Design_Document.pdf",
        "openapi.yaml",
    ):
        source = settings.documents_dir / name
        exists = source.is_file()
        print(f"{'OK' if exists else 'FAIL'}: source {source}")
        failed = failed or not exists

    if args.offline:
        print("Ollama connection: skipped (--offline).")
        return 1 if failed else 0

    endpoint = f"{str(settings.ollama_base_url).rstrip('/')}/api/tags"
    try:
        with urlopen(endpoint, timeout=10) as response:
            payload = json.load(response)
        available = {entry["name"] for entry in payload["models"]}
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: Cannot read Ollama model list at {endpoint}: {exc}", file=sys.stderr)
        print("Start Ollama or run 'ollama serve', then retry.", file=sys.stderr)
        return 1

    for model in (settings.llm_model, settings.embedding_model):
        if model in available or (":" not in model and f"{model}:latest" in available):
            print(f"OK: installed model {model}")
        else:
            print(f"FAIL: Missing model {model}. Run: ollama pull {model}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
