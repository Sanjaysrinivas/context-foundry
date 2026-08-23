"""The visible ingestion and retrieval-augmented generation loop."""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace

from local_rag.documents import (
    chunk_pages,
    clean_extracted_markdown,
    flow_diagram_markdown,
    load_document,
)
from local_rag.domain import Answer, DocumentInfo, GroundedResponse, RAGError, SearchResult
from local_rag.providers import ChatProvider, EmbeddingProvider
from local_rag.store import VectorStore, lexical_tokens

QUERY_BOUNDARY_RE = re.compile(
    r"(?:[;?]\s+|,\s*(?=(?:(?:then|also)\s+)?"
    r"(?:explain|describe|compare|list|specify|identify|summarize|why|how|what|which|who|when|where)\b)"
    r"|\s+(?:and|then|also)\s+(?="
    r"(?:explain|describe|compare|list|specify|identify|summarize|why|how|what|which|who|when|where)\b))",
    re.IGNORECASE,
)
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
NO_EVIDENCE_RESPONSE = "I could not find enough relevant evidence in the indexed documents."
MIN_NEW_QUERY_TERMS = 2
TOP_CONTEXT_COVERAGE_THRESHOLD = 0.6
QUESTION_CONTROL_TERMS = {
    "complete",
    "describe",
    "every",
    "explain",
    "identify",
    "include",
    "list",
    "specify",
    "summarize",
    "trace",
}


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
                    self.top_k,
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
        evidence = _interleave_unique(groups, queries, self.top_k * len(queries))
        if not evidence:
            return Answer(NO_EVIDENCE_RESPONSE, [])

        citation_numbers = {
            _result_key(match): index for index, (match, _query) in enumerate(evidence, 1)
        }
        answers: list[str] = []
        for query, group in zip(queries, groups, strict=True):
            scoped = [
                match
                for match in _select_generation_context(query, group)
                if _result_key(match) in citation_numbers
            ]
            if not scoped:
                text = f"Insufficient evidence for: {query}"
            else:
                context = "\n\n".join(
                    f"[{index}] {match.source}, page {match.page}\n{_context_text(match.text)}"
                    for index, match in enumerate(scoped, 1)
                )
                draft = await self.chat_provider.answer(query, context, len(scoped))
                local_to_global = {
                    index: citation_numbers[_result_key(match)]
                    for index, match in enumerate(scoped, 1)
                }
                text = _render_grounded_response(draft, local_to_global, query)
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


def _select_generation_context(query: str, matches: list[SearchResult]) -> list[SearchResult]:
    """Keep the top result, then only passages that add meaningful query coverage."""
    query_terms = lexical_tokens(query) - QUESTION_CONTROL_TERMS
    if matches and query_terms:
        top_coverage = len(query_terms & lexical_tokens(matches[0].text)) / len(query_terms)
        if top_coverage >= TOP_CONTEXT_COVERAGE_THRESHOLD:
            return matches[:1]

    selected: list[SearchResult] = []
    covered: set[str] = set()
    for match in matches:
        matching_terms = query_terms & lexical_tokens(match.text)
        if not selected or len(matching_terms - covered) >= MIN_NEW_QUERY_TERMS:
            selected.append(match)
            covered.update(matching_terms)
    return selected


def _context_text(text: str) -> str:
    diagram = flow_diagram_markdown(text)
    return diagram or text


def _render_grounded_response(
    response: GroundedResponse, citation_mapping: dict[int, int], query: str
) -> str:
    claims: list[str] = []
    for claim in response.claims:
        citations = sorted({citation_mapping[number] for number in claim.citations})
        references = " ".join(f"[{number}]" for number in citations)
        claim_text = UNICODE_ESCAPE_RE.sub(
            lambda match: chr(int(match.group(1), 16)), claim.text.strip()
        )
        claims.append(f"{claim_text} {references}")

    if response.style == "steps":
        supported = "\n".join(f"{index}. {claim}" for index, claim in enumerate(claims, 1))
    elif response.style == "bullets":
        supported = "\n".join(f"- {claim}" for claim in claims)
    else:
        supported = "\n\n".join(claims)

    unsupported = "; ".join(item.strip() for item in response.unsupported if item.strip())
    if supported and unsupported:
        return f"{supported}\n\nInsufficient evidence for: {unsupported}"
    if supported:
        return supported
    return f"Insufficient evidence for: {unsupported or query}"
