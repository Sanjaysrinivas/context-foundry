"""The visible ingestion and retrieval-augmented generation loop."""

from __future__ import annotations

import asyncio
import re

from local_rag.documents import chunk_pages, load_document
from local_rag.domain import Answer, DocumentInfo, RAGError, SearchResult
from local_rag.providers import ChatProvider, EmbeddingProvider
from local_rag.store import VectorStore

QUERY_BOUNDARY_RE = re.compile(
    r"(?:[;?]\s+|,\s*(?:then|also)\s+|\s+and\s+"
    r"(?=(?:explain|describe|compare|list|specify|identify|summarize|why|how|what|which|who|when|where)\b))",
    re.IGNORECASE,
)


class RAGService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chat_provider: ChatProvider,
        store: VectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        score_threshold: float,
        embedding_batch_size: int = 32,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chat_provider = chat_provider
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._store_lock = asyncio.Lock()

    async def ingest(self, filename: str, content: bytes) -> DocumentInfo:
        pages = await asyncio.to_thread(load_document, filename, content)
        chunks = await asyncio.to_thread(chunk_pages, pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise RAGError("The document produced no searchable chunks")
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            embedded = await self.embedding_provider.embed([chunk.text for chunk in batch])
            if len(embedded) != len(batch):
                raise RAGError("Embedding provider returned the wrong number of vectors")
            vectors.extend(embedded)
        async with self._store_lock:
            await asyncio.to_thread(self.store.replace, chunks, vectors)
        return DocumentInfo(
            chunks[0].document_id,
            filename,
            len(chunks),
            len(pages),
            chunks[0].source_sha256,
        )

    async def retrieve(
        self, question: str, document_ids: list[str] | None = None
    ) -> list[SearchResult]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("Question must not be empty")
        queries = _retrieval_queries(clean_question)
        vectors = await self.embedding_provider.embed(queries)
        if len(vectors) != len(queries):
            raise RAGError("Embedding provider returned the wrong number of vectors")
        async with self._store_lock:
            matches = [
                await asyncio.to_thread(
                    self.store.search,
                    vector,
                    query,
                    self.top_k,
                    self.score_threshold,
                    document_ids,
                )
                for query, vector in zip(queries, vectors, strict=True)
            ]
        limit = self.top_k if len(queries) == 1 else self.top_k * 2
        return _interleave_unique(matches, limit)

    async def ask(self, question: str, document_ids: list[str] | None = None) -> Answer:
        clean_question = question.strip()
        matches = await self.retrieve(clean_question, document_ids)
        if not matches:
            return Answer(
                "I could not find enough relevant evidence in the indexed documents.",
                [],
            )

        context = "\n\n".join(
            f"[{index}] {match.source}, page {match.page}\n{match.text}"
            for index, match in enumerate(matches, 1)
        )
        text = await self.chat_provider.answer(clean_question, context)
        return Answer(text, matches)

    async def list_documents(self) -> list[DocumentInfo]:
        async with self._store_lock:
            return await asyncio.to_thread(self.store.list_documents)

    async def delete_document(self, document_id: str) -> bool:
        async with self._store_lock:
            return await asyncio.to_thread(self.store.delete_document, document_id)

    async def clear(self) -> None:
        async with self._store_lock:
            await asyncio.to_thread(self.store.clear)

    def close(self) -> None:
        self.store.close()


def _retrieval_queries(question: str) -> list[str]:
    parts = [part.strip(" ,.;:?") for part in QUERY_BOUNDARY_RE.split(question)]
    useful_parts = [part for part in parts if len(part.split()) >= 3]
    return useful_parts[:3] if len(useful_parts) > 1 else [question]


def _interleave_unique(groups: list[list[SearchResult]], limit: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[tuple[str, int, str]] = set()
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            match = group[rank]
            key = (match.document_id or match.source, match.page, match.text)
            if key in seen:
                continue
            seen.add(key)
            results.append(match)
            if len(results) == limit:
                return results
    return results
