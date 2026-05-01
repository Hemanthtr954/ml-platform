"""
Retrieval Layer — wraps ChromaDB retriever with optional hybrid search.

Search modes:
  similarity  — pure cosine/dot-product similarity (fast, may return duplicates)
  mmr         — Maximal Marginal Relevance (balances relevance vs diversity)
  hybrid      — BM25 keyword + MMR vector, merged via EnsembleRetriever

MMR is the default: it prevents returning five near-identical chunks when the
answer lives in slightly different phrasings across the same paragraph.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


@dataclass
class RetrievalConfig:
    k: int = 4                          # chunks to return
    fetch_k: int = 20                   # candidates for MMR re-ranking
    search_mode: SearchMode = SearchMode.MMR
    bm25_weight: float = 0.4            # only used in hybrid mode
    vector_weight: float = 0.6          # only used in hybrid mode
    metadata_filter: Optional[dict] = None


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: Optional[int]
    score: Optional[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float
    search_mode: str
    total_retrieved: int


class RAGRetriever:
    """
    Retrieval engine over a ChromaVectorStore.
    Supports similarity, MMR, and hybrid (BM25 + vector) search.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.time()

        if self.config.search_mode == SearchMode.HYBRID:
            docs = self._hybrid_retrieve(query)
        else:
            retriever = self.vector_store.as_retriever(
                k=self.config.k,
                search_type=self.config.search_mode.value,
                fetch_k=self.config.fetch_k,
                filter=self.config.metadata_filter,
            )
            docs = retriever.invoke(query)

        chunks = [
            RetrievedChunk(
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page"),
                score=d.metadata.get("score"),
                metadata=d.metadata,
            )
            for d in docs
        ]

        latency_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"[Retriever] '{query[:60]}...' → {len(chunks)} chunks "
            f"({self.config.search_mode}, {latency_ms}ms)"
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            latency_ms=latency_ms,
            search_mode=self.config.search_mode.value,
            total_retrieved=len(chunks),
        )

    def _hybrid_retrieve(self, query: str) -> list:
        """BM25 + vector search via LangChain EnsembleRetriever."""
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        # Fetch a larger candidate set from the vector store for BM25 corpus
        all_docs = self.vector_store.similarity_search(query, k=50)
        if not all_docs:
            return []

        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = self.config.k

        vector_retriever = self.vector_store.as_retriever(
            k=self.config.k,
            search_type="mmr",
            fetch_k=self.config.fetch_k,
        )

        ensemble = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[self.config.bm25_weight, self.config.vector_weight],
        )
        return ensemble.invoke(query)

    def format_context(self, result: RetrievalResult) -> str:
        parts = []
        for i, chunk in enumerate(result.chunks, 1):
            source_label = f"{chunk.source}"
            if chunk.page is not None:
                source_label += f", page {chunk.page}"
            parts.append(f"[{i}] Source: {source_label}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)
