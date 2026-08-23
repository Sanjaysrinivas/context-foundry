"""Vector store contract and persistent local Qdrant implementation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, cast

from qdrant_client import QdrantClient, models

from local_rag.domain import Chunk, DocumentInfo, RAGError, SearchResult

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOP_WORDS = {
    "and",
    "are",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "the",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class VectorStore(Protocol):
    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        query: str,
        limit: int,
        threshold: float,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]: ...

    def list_documents(self) -> list[DocumentInfo]: ...

    def delete_document(self, document_id: str) -> bool: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


class QdrantVectorStore:
    def __init__(self, path: Path, collection: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise RAGError("Chunks and vectors must be non-empty and have equal lengths")
        if (
            len({chunk.document_id for chunk in chunks}) != 1
            or len({chunk.source for chunk in chunks}) != 1
        ):
            raise RAGError("A document replacement must contain one source document")
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
                    "source_sha256": chunk.source_sha256,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchValue(value=chunks[0].source)
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=chunks[0].document_id)
                    )
                ],
            ),
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        query: str,
        limit: int,
        threshold: float,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        if not self.client.collection_exists(self.collection):
            return []
        query_filter = self._document_filter(document_ids)
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=max(64, limit * 8),
            with_payload=True,
        )
        query_tokens = _tokens(query)
        candidates: dict[str, tuple[dict[str, Any], float, float]] = {}
        for point in response.points:
            payload = cast(dict[str, Any], point.payload or {})
            candidates[str(point.id)] = (payload, max(0.0, float(point.score)), 0.0)

        if query_tokens:
            # ponytail: a payload scan is sufficient for a local, single-user corpus.
            # Replace it with Qdrant sparse vectors only when measured corpus scale requires it.
            for record in self._scroll(query_filter):
                payload = cast(dict[str, Any], record.payload or {})
                lexical_score = len(query_tokens & _tokens(str(payload.get("text", "")))) / len(
                    query_tokens
                )
                if lexical_score:
                    key = str(record.id)
                    _, dense_score, _ = candidates.get(key, (payload, 0.0, 0.0))
                    candidates[key] = (payload, dense_score, lexical_score)

        ranked: list[tuple[float, dict[str, Any]]] = []
        for payload, dense_score, lexical_score in candidates.values():
            if dense_score < threshold and lexical_score < 0.5:
                continue
            ranked.append((0.75 * dense_score + 0.25 * lexical_score, payload))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results: list[SearchResult] = []
        for score, payload in ranked[:limit]:
            results.append(
                SearchResult(
                    source=str(payload.get("source", "unknown")),
                    page=int(payload.get("page", 1)),
                    text=str(payload.get("text", "")),
                    score=score,
                    document_id=str(payload.get("document_id", "")),
                    source_sha256=str(payload.get("source_sha256", "")),
                )
            )
        return results

    def list_documents(self) -> list[DocumentInfo]:
        if not self.client.collection_exists(self.collection):
            return []
        grouped: dict[str, tuple[str, str, int, set[int]]] = {}
        for point in self._scroll():
            payload = cast(dict[str, Any], point.payload or {})
            document_id = str(payload.get("document_id", ""))
            source, source_sha256, chunks, pages = grouped.get(
                document_id,
                (
                    str(payload.get("source", "unknown")),
                    str(payload.get("source_sha256", "")),
                    0,
                    set(),
                ),
            )
            pages.add(int(payload.get("page", 1)))
            grouped[document_id] = (source, source_sha256, chunks + 1, pages)
        return sorted(
            (
                DocumentInfo(document_id, source, chunks, len(pages), source_sha256)
                for document_id, (source, source_sha256, chunks, pages) in grouped.items()
            ),
            key=lambda item: item.source.lower(),
        )

    def delete_document(self, document_id: str) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        selector = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )
        if not self.client.scroll(
            collection_name=self.collection,
            scroll_filter=selector,
            limit=1,
            with_payload=False,
        )[0]:
            return False
        self.client.delete(collection_name=self.collection, points_selector=selector, wait=True)
        return True

    def clear(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def close(self) -> None:
        self.client.close()

    def _ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            return
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        existing_dimension = vectors.size if isinstance(vectors, models.VectorParams) else None
        if existing_dimension != dimension:
            raise RAGError(
                "Embedding dimension changed. Clear the collection before using a "
                "different embedding model."
            )

    def _scroll(self, scroll_filter: models.Filter | None = None) -> list[models.Record]:
        records: list[models.Record] = []
        offset: Any = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                return records

    @staticmethod
    def _document_filter(document_ids: list[str] | None) -> models.Filter | None:
        if not document_ids:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
        )


def _tokens(text: str) -> set[str]:
    return set(_lexical_terms(text))


def _lexical_terms(text: str) -> list[str]:
    return [
        _singularize(token)
        for token in (match.group().lower() for match in TOKEN_RE.finditer(text))
        if len(token) > 2 and token not in STOP_WORDS
    ]


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("is", "ss", "us")):
        return token[:-1]
    return token
