from local_rag.domain import Chunk, DocumentInfo, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeChat:
    def __init__(self) -> None:
        self.context = ""

    async def answer(self, question: str, context: str) -> str:
        self.context = context
        return f"Grounded answer for {question} [1]"


class FakeStore:
    def __init__(self, matches: list[SearchResult] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self.matches = matches or []

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks = chunks

    def search(
        self,
        vector: list[float],
        query: str,
        limit: int,
        threshold: float,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        assert vector and limit == 4 and threshold == 0.25
        assert query
        return self.matches

    def list_documents(self) -> list[DocumentInfo]:
        if not self.chunks:
            return []
        chunk = self.chunks[0]
        return [
            DocumentInfo(
                chunk.document_id,
                chunk.source,
                len(self.chunks),
                1,
                chunk.source_sha256,
            )
        ]

    def delete_document(self, document_id: str) -> bool:
        found = bool(self.chunks and self.chunks[0].document_id == document_id)
        if found:
            self.chunks = []
        return found

    def clear(self) -> None:
        self.chunks = []

    def close(self) -> None:
        pass


def service(store: FakeStore, chat: FakeChat | None = None) -> RAGService:
    return RAGService(
        FakeEmbeddings(),
        chat or FakeChat(),
        store,
        chunk_size=30,
        chunk_overlap=5,
        top_k=4,
        score_threshold=0.25,
    )


async def test_ingest_embeds_and_stores_chunks() -> None:
    store = FakeStore()

    document = await service(store).ingest(
        "notes.txt", b"A local RAG keeps private documents private."
    )

    assert document.chunks == 2
    assert document.pages == 1
    assert len(store.chunks) == 2
    assert store.chunks[0].source == "notes.txt"
    assert document.source_sha256 == store.chunks[0].source_sha256
    assert document.document_id == document.source_sha256


async def test_answer_includes_retrieved_context() -> None:
    match = SearchResult("notes.txt", 1, "Private documents stay local.", 0.91)
    store = FakeStore([match])
    chat = FakeChat()

    result = await service(store, chat).ask("Where are documents stored?")

    assert result.citations == [match]
    assert "notes.txt, page 1" in chat.context
    assert result.text.endswith("[1]")


async def test_answer_skips_chat_without_evidence() -> None:
    result = await service(FakeStore()).ask("Unknown?")

    assert result.citations == []
    assert "could not find" in result.text


async def test_document_lifecycle() -> None:
    store = FakeStore()
    rag = service(store)
    document = await rag.ingest("notes.txt", b"Local evidence")

    assert await rag.list_documents() == [document]
    assert await rag.delete_document(document.document_id)
    assert await rag.list_documents() == []


async def test_ingestion_batches_embeddings() -> None:
    embeddings = FakeEmbeddings()
    store = FakeStore()
    rag = RAGService(
        embeddings,
        FakeChat(),
        store,
        chunk_size=20,
        chunk_overlap=0,
        embedding_batch_size=2,
        top_k=4,
        score_threshold=0.25,
    )

    document = await rag.ingest("notes.txt", b"one two three four five six seven eight nine ten")

    assert len(embeddings.batch_sizes) > 1
    assert max(embeddings.batch_sizes) == 2
    assert sum(embeddings.batch_sizes) == document.chunks
