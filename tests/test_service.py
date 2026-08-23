import re

from local_rag.domain import Chunk, DocumentInfo, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeChat:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.context = ""
        self.contexts: list[str] = []
        self.questions: list[str] = []
        self.responses = responses or []

    async def answer(self, question: str, context: str) -> str:
        self.context = context
        self.contexts.append(context)
        self.questions.append(question)
        if self.responses:
            return self.responses.pop(0)
        citation = re.search(r"\[(\d+)]", context)
        assert citation
        return f"Grounded answer for {question.splitlines()[0]} [{citation.group(1)}]"


class FakeStore:
    def __init__(
        self,
        matches: list[SearchResult] | None = None,
        matches_by_query: dict[str, list[SearchResult]] | None = None,
    ) -> None:
        self.chunks: list[Chunk] = []
        self.matches = matches or []
        self.matches_by_query = matches_by_query or {}
        self.queries: list[str] = []
        self.limits: list[int] = []

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
        assert vector and limit in {4, 8} and threshold == 0.25
        assert query
        self.queries.append(query)
        self.limits.append(limit)
        return self.matches_by_query.get(query, self.matches)

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


async def test_answer_cleans_and_sentence_splits_legacy_pdf_context() -> None:
    match = SearchResult(
        "report.pdf",
        9,
        "Use <mark>`chunk_id` s</mark>. Store the verified text.",
        0.91,
    )
    chat = FakeChat()

    result = await service(FakeStore([match]), chat).ask("What should be stored?")

    assert "<mark>" not in chat.context
    assert "`chunk_ids`.\nStore the verified text." in chat.context
    assert "<mark>" not in result.citations[0].text


async def test_answer_skips_chat_without_evidence() -> None:
    result = await service(FakeStore()).ask("Unknown?")

    assert result.citations == []
    assert "could not find" in result.text


async def test_retrieve_covers_compound_question_parts() -> None:
    validation = SearchResult("report.pdf", 8, "evidence support", 0.8, "doc")
    anchors = SearchResult("report.pdf", 9, "source hash and page coordinates", 0.7, "doc")
    chunk_ids = SearchResult("report.pdf", 9, "chunk IDs change", 0.6, "doc")
    duplicate = SearchResult("report.pdf", 1, "stable evidence", 0.56, "doc")
    queries = [
        "List every automatic validation check",
        "specify every stable evidence-anchor field",
        "explain why chunk IDs cannot be gold labels",
    ]
    store = FakeStore(
        matches_by_query={
            queries[0]: [validation, duplicate],
            queries[1]: [anchors, duplicate],
            queries[2]: [chunk_ids, duplicate],
        }
    )
    embeddings = FakeEmbeddings()
    chat = FakeChat()
    rag = RAGService(
        embeddings,
        chat,
        store,
        chunk_size=30,
        chunk_overlap=5,
        top_k=4,
        score_threshold=0.25,
    )

    answer = await rag.ask(
        "List every automatic validation check, specify every stable evidence-anchor "
        "field and explain why chunk IDs cannot be gold labels"
    )
    results = answer.citations

    assert store.queries == queries
    assert store.limits == [8, 8, 4]
    assert [result.text for result in results] == [
        "evidence support",
        "source hash and page coordinates",
        "chunk IDs change",
        "stable evidence",
    ]
    assert embeddings.batch_sizes == [3]
    assert chat.questions[:2] == queries[:2]
    assert chat.questions[2].startswith(queries[2])
    assert "every change condition" in chat.questions[2]
    assert chat.questions[2].endswith("then stop.")
    assert len(chat.contexts) == 3
    assert "evidence support" in chat.contexts[0]
    assert "source hash and page coordinates" in chat.contexts[1]
    assert "chunk IDs change" in chat.contexts[2]
    assert "## List every automatic validation check" in answer.text
    assert "## specify every stable evidence-anchor field" in answer.text
    assert f"Grounded answer for {queries[0]} [1]" in answer.text
    assert f"Grounded answer for {queries[1]} [2]" in answer.text
    assert f"Grounded answer for {queries[2]} [3]" in answer.text


async def test_answer_repairs_missing_inline_citations() -> None:
    match = SearchResult("notes.txt", 1, "Private documents stay local.", 0.91)
    chat = FakeChat(["Documents stay local.", "Documents stay local [1]."])

    result = await service(FakeStore([match]), chat).ask("Where are documents stored?")

    assert result.text == "Documents stay local [1]."
    assert len(chat.questions) == 2
    assert "Previous draft:\nDocuments stay local." in chat.questions[1]


async def test_answer_repairs_citation_heading_with_uncited_claims() -> None:
    match = SearchResult("notes.txt", 1, "Use a source hash.", 0.91)
    chat = FakeChat(["[1] Fields:\n- source hash", "Fields:\n- source hash [1]"])

    result = await service(FakeStore([match]), chat).ask("Which fields are stable?")

    assert result.text == "Fields:\n- source hash [1]"
    assert len(chat.questions) == 2


async def test_answer_repairs_uncited_claim_line_after_cited_line() -> None:
    match = SearchResult("notes.txt", 1, "Documents stay local.", 0.91)
    chat = FakeChat(
        [
            "Documents stay local [1].\nThey remain private.",
            "Documents stay local [1].\nThey remain private [1].",
        ]
    )

    result = await service(FakeStore([match]), chat).ask("Where are documents stored?")

    assert result.text == "Documents stay local [1].\nThey remain private [1]."
    assert len(chat.questions) == 2


async def test_answer_repairs_supported_claims_before_partial_abstention() -> None:
    match = SearchResult("notes.txt", 1, "Documents stay local.", 0.91)
    chat = FakeChat(
        [
            "Documents stay local.\n\nInsufficient evidence for: retention period",
            "Documents stay local [1].\n\nInsufficient evidence for: retention period",
        ]
    )

    result = await service(FakeStore([match]), chat).ask("Where are documents stored?")

    assert result.text.startswith("Documents stay local [1].")
    assert len(chat.questions) == 2


async def test_exhaustive_question_excludes_competing_unrelated_schema() -> None:
    direct = SearchResult("report.pdf", 9, "Evidence anchors use a source hash and page.", 0.8)
    generic_schema = SearchResult(
        "report.pdf",
        12,
        "Expected evidence: stable evidence anchors, reviewer, and model version.",
        0.79,
    )
    chat = FakeChat()

    await service(FakeStore([direct, generic_schema]), chat).ask(
        "Specify every stable evidence anchor field"
    )

    assert direct.text in chat.context
    assert generic_schema.text not in chat.context


async def test_list_every_prefers_complete_structured_list() -> None:
    overview = SearchResult(
        "report.pdf", 25, "Automatic validation is recommended.\nUse human review.", 0.9
    )
    complete = SearchResult(
        "report.pdf",
        8,
        "Automatic validation\n- evidence support\n- answerability\n- closed-book test",
        0.82,
    )
    chat = FakeChat()

    await service(FakeStore([overview, complete]), chat).ask(
        "List every automatic validation check"
    )

    assert complete.text in chat.context
    assert overview.text not in chat.context


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
