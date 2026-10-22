"""Terminal entry points. Only CLI startup loads user-managed configuration."""

import argparse
import sys
from contextlib import closing

from pydantic import ValidationError

from freezer_rag.application import ChatSession
from freezer_rag.config import Settings
from freezer_rag.embeddings.ollama import Embeddings
from freezer_rag.models import Answer, RagError
from freezer_rag.ollama import Ollama
from freezer_rag.vectorstore.store import IndexStore, store_owner


def print_sources(answer: Answer | None) -> None:
    if answer is None or not answer.sources:
        print("No sources for the last completed answer.")
        return
    print("\nSources:")
    for label, chunk in answer.sources.items():
        print(f"[{label}] {chunk.source} - {chunk.locator}")


def terminal_chat(session: ChatSession) -> None:
    print("Freezer document chat. /help for commands. Ctrl+C cancels; /exit closes.")
    while True:
        try:
            question = input("\nyou> ").strip()
            if not question:
                continue
            if question == "/exit":
                break
            if question == "/help":
                print("/help  /clear (reset history)  /sources (last answer)  /exit")
                continue
            if question == "/clear":
                session.clear()
                print("Conversation cleared.")
                continue
            if question == "/sources":
                print_sources(session.last_answer)
                continue
            if question.startswith("/"):
                print("Unknown command. Use /help.")
                continue
            print("Retrieving evidence and generating...", file=sys.stderr, flush=True)
            answer = session.ask(question)
            print(f"\nassistant> {answer.text}")
            print_sources(answer)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nCancelled. Enter another question or /exit.", file=sys.stderr)
        except RagError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local freezer-document RAG")
    root.add_argument("--no-env", action="store_true", help="Do not open/read a .env file")
    root.add_argument("--model", dest="llm_model", help="Override the answer model")
    root.add_argument(
        "--embedding-model", help="Override embeddings (selects a matching default index)"
    )
    root.add_argument("--collection", dest="qdrant_collection")
    root.add_argument("--temperature", dest="llm_temperature", type=float)
    root.add_argument("--num-ctx", dest="llm_num_ctx", type=int)
    root.add_argument("--num-predict", dest="llm_num_predict", type=int)
    root.add_argument("--dense-top-k", type=int)
    root.add_argument("--lexical-top-k", type=int)
    root.add_argument("--final-top-k", type=int)
    root.add_argument("--max-context-chars", type=int)
    root.add_argument("--embedding-keep-alive", help="Model residency during ingestion, e.g. 1m")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Build/update the document index")
    ingest.add_argument(
        "--rebuild", action="store_true", help="Force a fresh atomic index snapshot"
    )
    commands.add_parser("chat", help="Start an interactive terminal conversation")
    commands.add_parser("ui", help="Start the optional Streamlit document chat")
    commands.add_parser("doctor", help="Check model and index readiness without generating")
    commands.add_parser(
        "snapshots", help="List retained snapshots for the selected embedding model"
    )
    restore = commands.add_parser("restore", help="Activate a retained compatible index snapshot")
    restore.add_argument("snapshot")
    for command in ("ask", "search"):
        child = commands.add_parser(
            command, help="Answer once" if command == "ask" else "Inspect retrieval"
        )
        child.add_argument("question")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "ui":
        import importlib.util
        import subprocess
        from pathlib import Path

        if importlib.util.find_spec("streamlit") is None:
            print(
                "Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr
            )
            return 1
        try:
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(Path(__file__).with_name("ui_app.py")),
                    "--server.address=127.0.0.1",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                    "--",
                    *(sys.argv[1:] if argv is None else argv),
                ]
            )
        except KeyboardInterrupt:
            return 130
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in Settings.model_fields and value is not None
    }
    # --no-env is an explicit guarantee used by all agent/test invocations.
    try:
        settings = Settings(_env_file=None if args.no_env else ".env", **overrides)
    except ValidationError as exc:
        # Pydantic's default display includes input values; only show messages.
        print(
            "Invalid configuration: "
            + "; ".join(error["msg"] for error in exc.errors(include_input=False)),
            file=sys.stderr,
        )
        return 1
    ollama = Ollama(settings)
    store = None
    try:
        embeddings = Embeddings(ollama)
        with store_owner(settings.qdrant_path), closing(IndexStore(settings, embeddings)) as store:
            if args.command == "ingest":
                store.build(args.rebuild, progress=lambda message: print(message, file=sys.stderr))
                ollama.unload_embeddings()
            elif args.command == "snapshots":
                for manifest in store.snapshots():
                    print(
                        f"{manifest['snapshot']}  {manifest['created_at']}  "
                        f"{manifest['chunk_count']} chunks"
                    )
            elif args.command == "restore":
                manifest = store.restore(args.snapshot)
                print(f"Restored {manifest['snapshot']} ({manifest['chunk_count']} chunks)")
            else:
                if args.command in ("chat", "ask", "doctor"):
                    ollama.check_chat_model()
                store.open()
                if args.command == "doctor":
                    print(f"Ready: base model {settings.llm_model}")
                    print(
                        f"Ready: embeddings {settings.embedding_model}, "
                        f"{settings.embedding_dimensions} dimensions"
                    )
                    print(f"Ready: {len(store.chunks)} chunks; {settings.qdrant_collection}")
                    return 0
                session = ChatSession(store)
                if args.command == "chat":
                    ollama.model(settings.llm_model)
                    print(f"Ready: {settings.llm_model}; embeddings {settings.embedding_model}")
                    terminal_chat(session)
                elif args.command == "ask":
                    answer = session.ask(args.question)
                    print(answer.text)
                    print_sources(answer)
                else:
                    for index, chunk in enumerate(session.search(args.question), 1):
                        print(f"\n[S{index}] {chunk.source} - {chunk.locator}\n{chunk.text}")
        return 0
    except KeyboardInterrupt:
        print("Cancelled; any previously active index remains available.", file=sys.stderr)
        return 130
    except (RagError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
        ollama.close()


if __name__ == "__main__":
    raise SystemExit(main())
