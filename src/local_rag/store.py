"""Vector store contract and persistent local Qdrant implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from qdrant_client import QdrantClient, models

from local_rag.domain import Chunk, RAGError, SearchResult


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], limit: int, threshold: float
    ) -> list[SearchResult]: ...

    def clear(self) -> None: ...


class QdrantVectorStore:
    def __init__(self, path: Path, collection: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise RAGError(
                "Chunks and vectors must be non-empty and have equal lengths"
            )
        dimension = len(vectors[0])
        if dimension == 0 or any(len(vector) != dimension for vector in vectors):
            raise RAGError("Embedding vectors must share a non-zero dimension")
        self._ensure_collection(dimension)
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self, vector: list[float], limit: int, threshold: float
    ) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection):
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            score_threshold=threshold,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = cast(dict[str, Any], point.payload or {})
            results.append(
                SearchResult(
                    source=str(payload.get("source", "unknown")),
                    page=int(payload.get("page", 1)),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return results

    def clear(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def _ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
            return
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        existing_dimension = (
            vectors.size if isinstance(vectors, models.VectorParams) else None
        )
        if existing_dimension != dimension:
            raise RAGError(
                "Embedding dimension changed. Clear the collection before using a different embedding model."
            )
