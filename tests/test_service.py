from local_rag.domain import Chunk, DocumentInfo, GroundedClaim, GroundedResponse, SearchResult
from local_rag.service import NO_EVIDENCE_RESPONSE, RAGService


def claim(text: str, *citations: int) -> GroundedClaim:
    return GroundedClaim(text=text, citations=list(citations or (1,)))


class FakeEmbeddings:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeChat:
    def __init__(self, responses: list[GroundedResponse] | None = None) -> None:
        self.contexts: list[str] = []
        self.questions: list[str] = []
        self.citation_counts: list[int] = []
        self.responses = responses or []

    @property
    def context(self) -> str:
        return self.contexts[-1]

    async def answer(self, question: str, context: str, citation_count: int) -> GroundedResponse:
        self.contexts.append(context)
        self.questions.append(question)
        self.citation_counts.append(citation_count)
        if self.responses:
            return self.responses.pop(0)
        return GroundedResponse(claims=[claim(f"Grounded answer for {question}")])


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


async def test_answer_renders_schema_claims_and_citations() -> None:
    match = SearchResult("notes.txt", 1, "Private documents stay local.", 0.91)
    chat = FakeChat(
        [
            GroundedResponse(
                style="bullets",
                claims=[
                    claim("- Documents stay local"),
                    claim("Documents remain private"),
                ],
            )
        ]
    )

    result = await service(FakeStore([match]), chat).ask("Where are documents stored?")

    assert result.citations == [match]
    assert "notes.txt, page 1" in chat.context
    assert chat.citation_counts == [1]
    assert result.text == "- Documents stay local [1]\n- Documents remain private [1]"


async def test_answer_renders_ordered_steps_and_partial_abstention() -> None:
    match = SearchResult("report.pdf", 8, "Collect evidence then review it.", 0.91)
    chat = FakeChat(
        [
            GroundedResponse(
                style="steps",
                claims=[
                    claim("Collect evidence"),
                    claim("Review it"),
                ],
                unsupported=["deployment decision"],
            )
        ]
    )

    result = await service(FakeStore([match]), chat).ask("Trace the process")

    assert result.text == (
        "1. Collect evidence [1]\n2. Review it [1]\n\n"
        "Insufficient evidence for: deployment decision"
    )


async def test_answer_decodes_literal_json_unicode_escapes() -> None:
    match = SearchResult("report.pdf", 8, "Next gold version", 0.91)
    chat = FakeChat([GroundedResponse(claims=[claim(r"Silver \u2192 gold")])])

    result = await service(FakeStore([match]), chat).ask("What comes next?")

    assert result.text == "Silver → gold [1]"


async def test_answer_cleans_and_formats_pdf_context() -> None:
    match = SearchResult(
        "report.pdf",
        9,
        "```\nCollect evidence\n  ↓\nExtract facts\n  ↓\nValidate\n"
        "  ├─ support\n  └─ answerability\n\nUse <mark>`chunk_id` s</mark>.",
        0.91,
    )
    chat = FakeChat()

    result = await service(FakeStore([match]), chat).ask("Trace the lifecycle")

    assert "1. Collect evidence" in chat.context
    assert "3. Validate\n   - support\n   - answerability" in chat.context
    assert "```" not in chat.context
    assert "<mark>" not in result.citations[0].text


async def test_answer_adds_context_that_contributes_missing_query_terms() -> None:
    first = SearchResult("report.pdf", 26, "Generated cases are synthetic silver.", 0.91)
    second = SearchResult("report.pdf", 27, "Human review can approve, edit, or reject.", 0.82)
    chat = FakeChat()

    await service(FakeStore([first, second]), chat).ask(
        "Explain synthetic silver and human review options"
    )

    assert first.text in chat.context
    assert second.text in chat.context
    assert chat.citation_counts == [2]


async def test_answer_uses_top_context_when_it_covers_most_question_concepts() -> None:
    complete = SearchResult(
        "report.pdf",
        8,
        "Coverage-aware sampling leads through validation, human review, frozen dataset "
        "release, evaluation, and the next gold version.",
        1.0,
    )
    alternate = SearchResult(
        "report.pdf", 7, "A different architecture lifecycle has evidence layers.", 0.92
    )
    chat = FakeChat()

    await service(FakeStore([complete, alternate]), chat).ask(
        "Trace the complete lifecycle from coverage-aware sampling to the next gold dataset "
        "version, including validation, human review, frozen release, and evaluation layers"
    )

    assert complete.text in chat.context
    assert alternate.text not in chat.context
    assert chat.citation_counts == [1]


async def test_answer_does_not_overfeed_redundant_retrieved_context() -> None:
    complete = SearchResult(
        "report.pdf",
        8,
        "Coverage-aware evidence sampling, automatic validation, human review, "
        "frozen dataset release, evaluation, and next gold version.",
        1.0,
    )
    related = SearchResult(
        "report.pdf",
        1,
        "Architecture uses evidence sampling, validation, review, release, and evaluation.",
        0.97,
    )
    chat = FakeChat()

    result = await service(FakeStore([complete, related]), chat).ask(
        "Trace coverage-aware evidence sampling through automatic validation, human review, "
        "frozen dataset release, evaluation, and the next gold version"
    )

    assert complete.text in chat.context
    assert related.text not in chat.context
    assert chat.citation_counts == [1]
    assert result.citations == [complete, related]


async def test_exhaustive_answer_prefers_deep_direct_definition() -> None:
    generic = [
        SearchResult(
            "report.pdf", page, f"Stable evidence anchor field table {page}.", 1 - page / 100
        )
        for page in range(1, 5)
    ]
    definition = SearchResult(
        "report.pdf",
        9,
        "Evidence anchors should use source hash + source page + coordinates + fingerprint.",
        0.90,
    )
    chat = FakeChat()

    result = await service(FakeStore([*generic, definition]), chat).ask(
        "Specify every field required for a stable evidence anchor"
    )

    assert definition.text in chat.context
    assert generic[0].text not in chat.context
    assert result.citations[0] == definition
    assert chat.citation_counts == [1]


async def test_answer_skips_chat_without_evidence() -> None:
    chat = FakeChat()

    result = await service(FakeStore(), chat).ask("Unknown?")

    assert result.text == NO_EVIDENCE_RESPONSE
    assert result.citations == []
    assert not chat.questions


async def test_retrieve_and_answer_cover_compound_question_parts() -> None:
    validation = SearchResult("report.pdf", 8, "evidence support", 0.8, "doc")
    anchors = SearchResult("report.pdf", 9, "source hash and page coordinates", 0.7, "doc")
    chunk_ids = SearchResult("report.pdf", 10, "chunk IDs change", 0.6, "doc")
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
    chat = FakeChat(
        [
            GroundedResponse(claims=[claim("Validation")]),
            GroundedResponse(claims=[claim("Anchors")]),
            GroundedResponse(claims=[claim("Chunk IDs change")]),
        ]
    )

    answer = await service(store, chat).ask(
        "List every automatic validation check, specify every stable evidence-anchor "
        "field and explain why chunk IDs cannot be gold labels"
    )

    assert store.queries == queries
    assert store.limits == [8, 8, 4]
    assert [result.text for result in answer.citations] == [
        "evidence support",
        "source hash and page coordinates",
        "chunk IDs change",
        "stable evidence",
    ]
    assert chat.questions == queries
    assert "Validation [1]" in answer.text
    assert "Anchors [2]" in answer.text
    assert "Chunk IDs change [3]" in answer.text


async def test_document_lifecycle() -> None:
    store = FakeStore()
    rag = service(store)
    document = await rag.ingest("notes.txt", b"Local evidence")

    assert await rag.list_documents() == [document]
    assert await rag.delete_document(document.document_id)
    assert await rag.list_documents() == []
    await rag.clear()
    rag.close()


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
