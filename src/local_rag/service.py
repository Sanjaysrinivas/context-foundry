"""The visible ingestion and retrieval-augmented generation loop."""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace

from local_rag.documents import chunk_pages, clean_extracted_markdown, load_document
from local_rag.domain import Answer, DocumentInfo, RAGError, SearchResult
from local_rag.providers import ChatProvider, EmbeddingProvider
from local_rag.store import VectorStore, _lexical_terms

QUERY_BOUNDARY_RE = re.compile(
    r"(?:[;?]\s+|,\s*(?=(?:(?:then|also)\s+)?"
    r"(?:explain|describe|compare|list|specify|identify|summarize|why|how|what|which|who|when|where)\b)"
    r"|\s+(?:and|then|also)\s+(?="
    r"(?:explain|describe|compare|list|specify|identify|summarize|why|how|what|which|who|when|where)\b))",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(\d+)]")
TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+$")
CONTEXT_SCORE_WINDOW = 0.07
DEFINITION_TERMS = {"define", "include", "require", "should", "use"}
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*])")


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
        queries, groups = await self._search_groups(question, document_ids)
        evidence = _interleave_unique(groups, queries, self.top_k * len(queries))
        return [match for match, _query in evidence]

    async def _search_groups(
        self, question: str, document_ids: list[str] | None = None
    ) -> tuple[list[str], list[list[SearchResult]]]:
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
                    self.top_k * 2 if "every" in query.casefold().split() else self.top_k,
                    self.score_threshold,
                    document_ids,
                )
                for query, vector in zip(queries, vectors, strict=True)
            ]
        return queries, [
            [replace(match, text=clean_extracted_markdown(match.text)) for match in group]
            for group in matches
        ]

    async def ask(self, question: str, document_ids: list[str] | None = None) -> Answer:
        clean_question = question.strip()
        queries, groups = await self._search_groups(clean_question, document_ids)
        scoped_groups = [
            _generation_matches(query, group) for query, group in zip(queries, groups, strict=True)
        ]
        evidence = _interleave_unique(scoped_groups, queries, self.top_k * len(queries))
        if not evidence:
            return Answer(
                "I could not find enough relevant evidence in the indexed documents.",
                [],
            )

        citation_numbers = {
            _result_key(match): index for index, (match, _query) in enumerate(evidence, 1)
        }
        answers: list[str] = []
        for query, group in zip(queries, scoped_groups, strict=True):
            scoped = [match for match in group if _result_key(match) in citation_numbers]
            if not scoped:
                text = f"Insufficient evidence for: {query}"
            else:
                context = "\n\n".join(
                    f"[{index}] {match.source}, page {match.page}\n{_context_text(match.text)}"
                    for index, match in enumerate(scoped, 1)
                )
                valid_citations = set(range(1, len(scoped) + 1))
                generation_question = _generation_question(query)
                text = await self.chat_provider.answer(generation_question, context)
                if _needs_citation_repair(text, valid_citations):
                    text = await self.chat_provider.answer(
                        f"{generation_question}\n\n"
                        "Rewrite the previous draft using only the evidence. "
                        "End every supported sentence or list item with at least one valid [n] "
                        "citation and return only the revised answer.\n\n"
                        f"Previous draft:\n{text}",
                        context,
                    )
                if _needs_citation_repair(text, valid_citations):
                    text = f"Insufficient evidence for: {query}"
                else:
                    local_to_global = {
                        index: citation_numbers[_result_key(match)]
                        for index, match in enumerate(scoped, 1)
                    }
                    text = _remap_citations(text, local_to_global)
            answers.append(f"## {query}\n\n{text}" if len(queries) > 1 else text)

        text = "\n\n".join(answers)
        matches = [match for match, _query in evidence]
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


def _interleave_unique(
    groups: list[list[SearchResult]], queries: list[str], limit: int
) -> list[tuple[SearchResult, str]]:
    results: list[tuple[SearchResult, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for rank in range(max((len(group) for group in groups), default=0)):
        for group, query in zip(groups, queries, strict=True):
            if rank >= len(group):
                continue
            match = group[rank]
            key = _result_key(match)
            if key in seen:
                continue
            seen.add(key)
            results.append((match, query))
            if len(results) == limit:
                return results
    return results


def _result_key(match: SearchResult) -> tuple[str, int, str]:
    return (match.document_id or match.source, match.page, match.text)


def _context_text(text: str) -> str:
    return SENTENCE_BREAK_RE.sub("\n", text)


def _generation_question(query: str) -> str:
    if query.casefold().startswith(("explain ", "why ", "how ")):
        return f"{query}\nInclude every change condition stated as a reason, then stop."
    return query


def _generation_matches(query: str, group: list[SearchResult]) -> list[SearchResult]:
    if not group:
        return []
    score_window = 0.1 if "every" in query.casefold().split() else CONTEXT_SCORE_WINDOW
    minimum_score = group[0].score - score_window
    competitive = [match for match in group if match.score >= minimum_score]
    if "every" not in query.casefold().split():
        return competitive

    query_terms = _lexical_terms(query)
    query_bigrams = set(zip(query_terms, query_terms[1:], strict=False))
    direct = [match for match in competitive if query_bigrams & _text_bigrams(match.text)]
    if query.casefold().lstrip().startswith("list "):
        candidates = direct or competitive
        most_complete = max(candidates, key=lambda match: len(match.text.splitlines()))
        if len(most_complete.text.splitlines()) >= 4:
            return [most_complete]
    definitions = [
        match for match in direct if _defines_requested_phrase(query_bigrams, match.text)
    ]
    return definitions or direct or competitive


def _text_bigrams(text: str) -> set[tuple[str, str]]:
    terms = _lexical_terms(text)
    return set(zip(terms, terms[1:], strict=False))


def _defines_requested_phrase(query_bigrams: set[tuple[str, str]], text: str) -> bool:
    terms = _lexical_terms(text)
    for index in range(len(terms) - 1):
        if (terms[index], terms[index + 1]) not in query_bigrams:
            continue
        nearby = terms[max(0, index - 3) : index + 7]
        if DEFINITION_TERMS.intersection(nearby):
            return True
    return False


def _needs_citation_repair(text: str, valid_citations: set[int]) -> bool:
    claim_lines: list[str] = []
    lines = [line.strip() for line in text.splitlines()]
    for index, stripped in enumerate(lines):
        lowered = stripped.casefold()
        is_heading = stripped.endswith(":") and not any(
            character in stripped[:-1] for character in ".!?"
        )
        is_table_structure = bool(TABLE_SEPARATOR_RE.fullmatch(stripped)) or (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and bool(TABLE_SEPARATOR_RE.fullmatch(lines[index + 1]))
        )
        if (
            not stripped
            or stripped.startswith("#")
            or is_heading
            or is_table_structure
            or lowered.startswith("insufficient evidence for:")
            or lowered == "i could not find enough relevant evidence in the indexed documents."
        ):
            continue
        claim_lines.append(stripped)
    if not claim_lines:
        return False
    citations = {int(value) for value in CITATION_RE.findall("\n".join(claim_lines))}
    if not citations or not citations.issubset(valid_citations):
        return True
    for claim in claim_lines:
        claim_citations = {int(value) for value in CITATION_RE.findall(claim)}
        if not claim_citations.intersection(valid_citations):
            return True
    return False


def _remap_citations(text: str, mapping: dict[int, int]) -> str:
    return CITATION_RE.sub(lambda match: f"[{mapping[int(match.group(1))]}]", text)
