"""Immutable index snapshots with an atomic manifest switch and process ownership."""

import json
import os
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models

from freezer_rag.config import Settings
from freezer_rag.embeddings.ollama import Embeddings
from freezer_rag.ingestion.chunking import PIPELINE_VERSION, chunk_units
from freezer_rag.ingestion.corpus import load_corpus, source_hashes
from freezer_rag.models import Chunk, RagError


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def store_owner(root: Path):
    """Use SQLite's OS-released transaction lock, avoiding stale PID/lock-file guessing."""
    import sqlite3

    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "ownership.sqlite", timeout=0, check_same_thread=False)
    try:
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise RagError(
                "This local index is already in use. Close other chat/ingestion processes first."
            ) from exc
        yield
    finally:
        connection.close()


class IndexStore:
    def __init__(self, settings: Settings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.pointer = settings.manifests_dir / f"{settings.qdrant_collection}.json"
        self.client: QdrantClient | None = None
        self.chunks: list[Chunk] = []

    def signature(self) -> dict:
        settings = self.settings
        return {
            "pipeline_version": PIPELINE_VERSION,
            "sources": source_hashes(settings.documents_dir),
            "embedding_model": settings.embedding_model,
            "embedding_digest": self.embeddings.digest,
            "prefix_strategy": self.embeddings.strategy,
            "dimensions": settings.embedding_dimensions,
            "distance": "Cosine",
            "collection": settings.qdrant_collection,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
        }

    def read_manifest(self) -> dict:
        if not self.pointer.is_file():
            raise RagError("No compatible index exists. Run: freezer-rag ingest")
        try:
            manifest = json.loads(self.pointer.read_text(encoding="utf-8"))
            if manifest["signature"] != self.signature():
                raise RagError(
                    "Index is stale or incompatible with sources/model/chunking. Run ingest."
                )
            snapshot = manifest["snapshot"]
            if (
                not isinstance(snapshot, str)
                or len(snapshot) != 32
                or not all(letter in "0123456789abcdef" for letter in snapshot)
            ):
                raise RagError("Invalid index snapshot locator; run ingest --rebuild")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RagError("Cannot read the index manifest. Run ingest --rebuild.") from exc

    def build(self, rebuild: bool = False, progress=print) -> dict:
        signature = self.signature()
        if self.pointer.exists() and not rebuild:
            try:
                manifest = self.read_manifest()
                # Also verify files/count; an interrupted/damaged snapshot must not look ready.
                self.open()
                self.close()
                progress("Index is already current; no documents were re-embedded.")
                return manifest
            except RagError:
                self.close()
        snapshot = uuid4().hex
        directory = self.settings.qdrant_path / "snapshots" / snapshot
        directory.mkdir(parents=True)
        progress("Extracting the three allowlisted sources...")
        units = load_corpus(self.settings.documents_dir, signature["sources"])
        # Limit bytes conservatively as well as characters, leaving room for prefixes.
        size = min(self.settings.chunk_size_chars, self.embeddings.input_limit - 192)
        if size < 128:
            raise RagError("Embedding model input limit is too small for document ingestion")
        chunks = chunk_units(units, size, min(self.settings.chunk_overlap_chars, size // 3))
        if not chunks:
            raise RagError("No chunks extracted; the active index was not changed")
        write_json(directory / "units.json", [unit.model_dump() for unit in units])
        client = QdrantClient(path=str(directory / "qdrant"))
        try:
            collection = self.settings.qdrant_collection
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            batch_size = self.settings.embedding_batch_size
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                vectors = self.embeddings.embed([chunk.normalized_text for chunk in batch])
                client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump())
                        for chunk, vector in zip(batch, vectors, strict=True)
                    ],
                )
                if offset == 0 or (offset // batch_size) % 10 == 0:
                    progress(
                        f"Embedded {min(offset + batch_size, len(chunks))}/{len(chunks)} chunks"
                    )
            if client.count(collection, exact=True).count != len(chunks):
                raise RagError("Index record count mismatch; the previous index remains active")
        finally:
            client.close()
        # JSON is also the persisted BM25 corpus; no executable pickle is loaded.
        write_json(directory / "chunks.json", [chunk.model_dump() for chunk in chunks])
        if source_hashes(self.settings.documents_dir) != signature["sources"]:
            raise RagError(
                "Sources changed during ingestion; retry. Previous index remains active."
            )
        from datetime import UTC, datetime

        manifest = {
            "snapshot": snapshot,
            "signature": signature,
            "chunk_count": len(chunks),
            "created_at": datetime.now(UTC).isoformat(),
        }
        write_json(directory / "manifest.json", manifest)
        # Human-inspectable exports; snapshots above remain the authoritative atomic bundle.
        write_json(
            self.settings.processed_dir / f"{collection}.json",
            {
                "snapshot": snapshot,
                "chunks": [chunk.model_dump() for chunk in chunks],
            },
        )
        # Only this operation activates the fully written/closed replacement.
        write_json(self.pointer, manifest)
        progress(f"Ready: {len(chunks)} chunks in {collection}")
        return manifest

    def open(self, *, shared: bool = False) -> None:
        manifest = self.read_manifest()
        directory = self.settings.qdrant_path / "snapshots" / manifest["snapshot"]
        try:
            self.chunks = [
                Chunk.model_validate(value)
                for value in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
            ]
            if len(self.chunks) != manifest["chunk_count"]:
                raise RagError("Index corpus is incomplete. Run ingest --rebuild.")
            if not (directory / "qdrant").is_dir():
                raise RagError("Vector index is missing. Run ingest --rebuild.")
            self.client = QdrantClient(
                path=str(directory / "qdrant"), force_disable_check_same_thread=shared
            )
            if self.client.count(self.settings.qdrant_collection, exact=True).count != len(
                self.chunks
            ):
                raise RagError("Vector index is incomplete. Run ingest --rebuild.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.close()
            raise RagError(f"Cannot open the index: {exc}. Run ingest --rebuild.") from exc

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def snapshots(self) -> list[dict]:
        snapshots = []
        root = self.settings.qdrant_path / "snapshots"
        for path in sorted(root.glob("*/manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest["signature"]["collection"] == self.settings.qdrant_collection:
                    snapshots.append(manifest)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return snapshots

    def restore(self, snapshot: str) -> dict:
        """Activate an existing compatible, complete snapshot without modifying its files."""
        if len(snapshot) != 32 or any(char not in "0123456789abcdef" for char in snapshot):
            raise RagError("Invalid snapshot ID; choose one from the snapshots command")
        directory = self.settings.qdrant_path / "snapshots" / snapshot
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["snapshot"] != snapshot or manifest["signature"] != self.signature():
                raise RagError("Snapshot is incompatible with current sources/model/chunking")
            chunks = [
                Chunk.model_validate(value)
                for value in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
            ]
            if len(chunks) != manifest["chunk_count"] or not (directory / "qdrant").is_dir():
                raise RagError("Snapshot is incomplete; active index was not changed")
            with closing(QdrantClient(path=str(directory / "qdrant"))) as candidate:
                if candidate.count(self.settings.qdrant_collection, exact=True).count != len(
                    chunks
                ):
                    raise RagError("Snapshot vectors are incomplete; active index was not changed")
            write_json(self.pointer, manifest)
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RagError("Cannot restore this snapshot; active index was not changed") from exc

    def dense(self, vector: list[float], limit: int) -> list[str]:
        if self.client is None:
            raise RagError("The vector index is not open")
        return [
            str(point.id)
            for point in self.client.query_points(
                self.settings.qdrant_collection,
                query=vector,
                limit=limit,
                with_payload=False,
                with_vectors=False,
            ).points
        ]
