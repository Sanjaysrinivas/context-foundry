"""The visible ingestion and retrieval-augmented generation loop."""

from __future__ import annotations

from local_rag.documents import chunk_pages, load_document
from local_rag.domain import Answer, RAGError
from local_rag.providers import ChatProvider, EmbeddingProvider
from local_rag.store import VectorStore


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
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chat_provider = chat_provider
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def ingest(self, filename: str, content: bytes) -> int:
        pages = load_document(filename, content)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise RAGError("The document produced no searchable chunks")
        vectors = await self.embedding_provider.embed([chunk.text for chunk in chunks])
        self.store.upsert(chunks, vectors)
        return len(chunks)

    async def ask(self, question: str) -> Answer:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("Question must not be empty")
        vectors = await self.embedding_provider.embed([clean_question])
        if len(vectors) != 1:
            raise RAGError("Embedding provider returned the wrong number of vectors")
        matches = self.store.search(vectors[0], self.top_k, self.score_threshold)
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

    def clear(self) -> None:
        self.store.clear()

    def close(self) -> None:
        self.store.close()
