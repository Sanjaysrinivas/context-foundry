from local_rag.domain import Chunk, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
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

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks = chunks

    def search(
        self, vector: list[float], limit: int, threshold: float
    ) -> list[SearchResult]:
        assert vector and limit == 4 and threshold == 0.25
        return self.matches

    def clear(self) -> None:
        self.chunks = []


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

    count = await service(store).ingest(
        "notes.txt", b"A local RAG keeps private documents private."
    )

    assert count == 2
    assert len(store.chunks) == 2
    assert store.chunks[0].source == "notes.txt"


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
