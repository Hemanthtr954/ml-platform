"""
RAG Platform API — FastAPI interface for the full RAG pipeline.

Endpoints:
  Ingest:
    POST /ingest/directory   — load & index files from a local directory
    POST /ingest/urls        — scrape & index web pages
    GET  /ingest/stats       — vectorstore document count and config

  Query:
    POST /query              — single-turn RAG query (returns answer + sources)
    POST /query/stream       — streaming RAG query (Server-Sent Events)
    POST /query/retrieve     — retrieve chunks only (no LLM)

  Evaluation:
    POST /eval               — evaluate RAG quality over a QA dataset

  Health:
    GET  /health             — liveness check
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

vector_store = None
retriever = None
rag_chain = None
evaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, retriever, rag_chain, evaluator

    from vectorstore.chroma_store import ChromaVectorStore, VectorStoreConfig, EmbeddingModel
    from retrieval.retriever import RAGRetriever, RetrievalConfig, SearchMode
    from chain.rag_chain import RAGChain, LLMConfig, LLMBackend
    from evals.rag_eval import RAGEvaluator

    vs_config = VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PATH", "./vectorstore_data"),
        embedding_model=EmbeddingModel(
            os.getenv("EMBEDDING_MODEL", EmbeddingModel.MINI_LM.value)
        ),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )
    vector_store = ChromaVectorStore(vs_config)

    retriever_config = RetrievalConfig(
        k=int(os.getenv("RETRIEVAL_K", "4")),
        search_mode=SearchMode(os.getenv("SEARCH_MODE", SearchMode.MMR.value)),
    )
    retriever = RAGRetriever(vector_store, retriever_config)

    llm_config = LLMConfig(
        backend=LLMBackend(os.getenv("LLM_BACKEND", LLMBackend.OLLAMA.value)),
        model_name=os.getenv("LLM_MODEL", "mistral"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        hf_api_token=os.getenv("HF_API_TOKEN"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
    )
    rag_chain = RAGChain(retriever, llm_config)
    evaluator = RAGEvaluator(pass_threshold=float(os.getenv("EVAL_PASS_THRESHOLD", "0.6")))

    logger.info(
        f"RAG Platform started | LLM={llm_config.backend}/{llm_config.model_name} "
        f"| Embeddings={vs_config.embedding_model.value}"
    )
    yield
    logger.info("RAG Platform shutting down")


app = FastAPI(
    title="RAG Platform",
    description="Production RAG pipeline: document ingest, vector search, LLM generation, evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request / Response Models ──────────────────────────────────────────────────

class IngestDirectoryRequest(BaseModel):
    directory: str
    rebuild: bool = False           # if True, drop existing store and rebuild
    chunk_size: int = 512
    chunk_overlap: int = 64

class IngestURLsRequest(BaseModel):
    urls: list[str]
    chunk_size: int = 512
    chunk_overlap: int = 64

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = None         # override retrieval k for this query
    search_mode: Optional[str] = None

class RetrieveRequest(BaseModel):
    question: str
    k: int = 4
    search_mode: str = "mmr"

class EvalRequest(BaseModel):
    qa_pairs: list[dict] = Field(
        description='[{"question": "...", "expected_answer": "..."}]'
    )

class EvalCaseRequest(BaseModel):
    question: str
    expected_answer: str
    retrieved_contexts: list[str]
    generated_answer: str


# ── Ingest Endpoints ──────────────────────────────────────────────────────────

@app.post("/ingest/directory")
async def ingest_directory(request: IngestDirectoryRequest):
    from ingest.loader import DocumentIngestor, IngestConfig

    config = IngestConfig(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    ingestor = DocumentIngestor(config)

    try:
        chunks, result = ingestor.ingest_directory(request.directory)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    if request.rebuild:
        count = vector_store.build_from_chunks(chunks)
    else:
        count = vector_store.add_documents(chunks)

    return {
        "status": "indexed",
        "documents_loaded": result.total_documents,
        "chunks_created": result.total_chunks,
        "total_in_store": count,
        "sources": result.sources,
        "elapsed_seconds": result.elapsed_seconds,
        "errors": result.errors,
    }


@app.post("/ingest/urls")
async def ingest_urls(request: IngestURLsRequest):
    from ingest.loader import DocumentIngestor, IngestConfig

    config = IngestConfig(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    ingestor = DocumentIngestor(config)
    chunks, result = ingestor.ingest_urls(request.urls)
    count = vector_store.add_documents(chunks)

    return {
        "status": "indexed",
        "urls_loaded": result.total_documents,
        "chunks_created": result.total_chunks,
        "total_in_store": count,
        "elapsed_seconds": result.elapsed_seconds,
        "errors": result.errors,
    }


@app.get("/ingest/stats")
async def ingest_stats():
    stats = vector_store.stats()
    return {
        "collection": stats.collection_name,
        "total_documents": stats.total_documents,
        "persist_directory": stats.persist_directory,
        "embedding_model": stats.embedding_model,
    }


# ── Query Endpoints ───────────────────────────────────────────────────────────

@app.post("/query")
async def query(request: QueryRequest):
    if request.k or request.search_mode:
        from retrieval.retriever import RetrievalConfig, SearchMode
        from chain.rag_chain import RAGChain

        cfg = RetrievalConfig(
            k=request.k or retriever.config.k,
            search_mode=SearchMode(request.search_mode) if request.search_mode else retriever.config.search_mode,
        )
        from retrieval.retriever import RAGRetriever
        temp_retriever = RAGRetriever(vector_store, cfg)
        chain = RAGChain(temp_retriever, rag_chain.llm_config)
        response = chain.query(request.question)
    else:
        response = rag_chain.query(request.question)

    return {
        "answer": response.answer,
        "query": response.query,
        "llm_backend": response.llm_backend,
        "model": response.model_name,
        "latency_ms": response.total_latency_ms,
        "retrieval_latency_ms": response.retrieval.latency_ms,
        "chunks_retrieved": response.retrieval.total_retrieved,
        "search_mode": response.retrieval.search_mode,
        "sources": response.sources,
    }


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    def token_generator():
        for token in rag_chain.stream(request.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/query/retrieve")
async def retrieve_only(request: RetrieveRequest):
    from retrieval.retriever import RetrievalConfig, SearchMode, RAGRetriever

    cfg = RetrievalConfig(k=request.k, search_mode=SearchMode(request.search_mode))
    temp_retriever = RAGRetriever(vector_store, cfg)
    result = temp_retriever.retrieve(request.question)

    return {
        "query": result.query,
        "chunks": [
            {
                "content": c.content,
                "source": c.source,
                "page": c.page,
                "metadata": c.metadata,
            }
            for c in result.chunks
        ],
        "latency_ms": result.latency_ms,
        "search_mode": result.search_mode,
    }


# ── Evaluation Endpoints ──────────────────────────────────────────────────────

@app.post("/eval")
async def run_eval(request: EvalRequest):
    if not request.qa_pairs:
        raise HTTPException(400, "qa_pairs must not be empty")

    report = evaluator.evaluate_from_rag_chain(rag_chain, request.qa_pairs)

    return {
        "total_cases": report.total_cases,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "pass_threshold": report.pass_threshold,
        "avg_context_recall": report.avg_context_recall,
        "avg_answer_relevance": report.avg_answer_relevance,
        "avg_faithfulness": report.avg_faithfulness,
        "avg_overall": report.avg_overall,
        "per_case": [
            {
                "context_recall": s.context_recall,
                "answer_relevance": s.answer_relevance,
                "faithfulness": s.faithfulness,
                "overall": s.overall,
            }
            for s in report.scores
        ],
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    store_count = 0
    try:
        store_count = vector_store.stats().total_documents
    except Exception:
        pass

    return {
        "status": "healthy",
        "components": {
            "vector_store": vector_store is not None,
            "retriever": retriever is not None,
            "rag_chain": rag_chain is not None,
            "evaluator": evaluator is not None,
        },
        "store_document_count": store_count,
        "llm_backend": rag_chain.llm_config.backend.value if rag_chain else None,
        "llm_model": rag_chain.llm_config.model_name if rag_chain else None,
    }
