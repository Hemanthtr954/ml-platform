"""
RAG Platform CLI

Usage:
  python main.py ingest  <directory>         — index all docs in a directory
  python main.py ingest  --url <url> [...]   — index web pages
  python main.py query   "<question>"        — ask a question (prints answer + sources)
  python main.py serve                       — start the FastAPI server
  python main.py eval    <qa_file.json>      — run evaluation on a QA dataset
  python main.py stats                       — show vectorstore stats

Environment variables (or .env file):
  LLM_BACKEND     = ollama | huggingface | hf_local | openai  (default: ollama)
  LLM_MODEL       = mistral (ollama) | mistralai/Mistral-7B-Instruct-v0.2 (hf)
  CHROMA_PATH     = ./vectorstore_data
  OLLAMA_BASE_URL = http://localhost:11434
  HF_API_TOKEN    = hf_...
  OPENAI_API_KEY  = sk-...
"""

from __future__ import annotations

import json
import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def cmd_ingest(args: list[str]):
    from ingest.loader import DocumentIngestor, IngestConfig
    from vectorstore.chroma_store import ChromaVectorStore, VectorStoreConfig

    config = IngestConfig()
    ingestor = DocumentIngestor(config)

    url_mode = "--url" in args
    rebuild = "--rebuild" in args
    filtered = [a for a in args if not a.startswith("--")]

    if url_mode:
        urls = filtered
        chunks, result = ingestor.ingest_urls(urls)
        print(f"Loaded {result.total_documents} pages → {result.total_chunks} chunks ({result.elapsed_seconds}s)")
    else:
        if not filtered:
            print("Error: provide a directory path")
            sys.exit(1)
        directory = filtered[0]
        chunks, result = ingestor.ingest_directory(directory)
        print(f"Loaded {result.total_documents} docs → {result.total_chunks} chunks ({result.elapsed_seconds}s)")
        if result.errors:
            print(f"Warnings: {result.errors}")

    vs = ChromaVectorStore(VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PATH", "./vectorstore_data")
    ))
    if rebuild:
        count = vs.build_from_chunks(chunks)
        print(f"Rebuilt store: {count} total chunks")
    else:
        count = vs.add_documents(chunks)
        print(f"Indexed. Store total: {count} chunks")


def cmd_query(args: list[str]):
    from vectorstore.chroma_store import ChromaVectorStore, VectorStoreConfig
    from retrieval.retriever import RAGRetriever, RetrievalConfig
    from chain.rag_chain import RAGChain, LLMConfig, LLMBackend

    if not args:
        print("Error: provide a question in quotes")
        sys.exit(1)

    question = " ".join(args)

    vs = ChromaVectorStore(VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PATH", "./vectorstore_data")
    ))
    ret = RAGRetriever(vs, RetrievalConfig())
    chain = RAGChain(
        ret,
        LLMConfig(
            backend=LLMBackend(os.getenv("LLM_BACKEND", "ollama")),
            model_name=os.getenv("LLM_MODEL", "mistral"),
        ),
    )

    print(f"\nQuestion: {question}\n")
    print("Thinking...\n")
    response = chain.query(question)

    print(f"Answer:\n{response.answer}\n")
    print(f"[{response.llm_backend}/{response.model_name} | {response.total_latency_ms}ms | {response.retrieval.total_retrieved} chunks]\n")
    print("Sources:")
    for s in response.sources:
        page = f" p.{s['page']}" if s["page"] is not None else ""
        print(f"  • {s['source']}{page}")
        print(f"    \"{s['excerpt'][:120]}...\"")


def cmd_stats():
    from vectorstore.chroma_store import ChromaVectorStore, VectorStoreConfig

    vs = ChromaVectorStore(VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PATH", "./vectorstore_data")
    ))
    stats = vs.stats()
    print(f"Collection : {stats.collection_name}")
    print(f"Documents  : {stats.total_documents}")
    print(f"Store path : {stats.persist_directory}")
    print(f"Embeddings : {stats.embedding_model}")


def cmd_eval(args: list[str]):
    if not args:
        print("Error: provide path to a QA JSON file")
        print('  Format: [{"question": "...", "expected_answer": "..."}]')
        sys.exit(1)

    qa_file = args[0]
    with open(qa_file) as f:
        qa_pairs = json.load(f)

    from vectorstore.chroma_store import ChromaVectorStore, VectorStoreConfig
    from retrieval.retriever import RAGRetriever, RetrievalConfig
    from chain.rag_chain import RAGChain, LLMConfig, LLMBackend
    from evals.rag_eval import RAGEvaluator

    vs = ChromaVectorStore(VectorStoreConfig(
        persist_directory=os.getenv("CHROMA_PATH", "./vectorstore_data")
    ))
    ret = RAGRetriever(vs, RetrievalConfig())
    chain = RAGChain(ret, LLMConfig(
        backend=LLMBackend(os.getenv("LLM_BACKEND", "ollama")),
        model_name=os.getenv("LLM_MODEL", "mistral"),
    ))
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_from_rag_chain(chain, qa_pairs)

    print(f"\n{'='*50}")
    print(f"RAG Evaluation Report — {report.total_cases} cases")
    print(f"{'='*50}")
    print(f"Context Recall   : {report.avg_context_recall:.3f}")
    print(f"Answer Relevance : {report.avg_answer_relevance:.3f}")
    print(f"Faithfulness     : {report.avg_faithfulness:.3f}")
    print(f"Overall Score    : {report.avg_overall:.3f}")
    print(f"Pass Rate        : {report.pass_rate:.1%} ({report.passed}/{report.total_cases})")
    print(f"{'='*50}")


def cmd_serve():
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


COMMANDS = {
    "ingest": cmd_ingest,
    "query": cmd_query,
    "stats": cmd_stats,
    "eval": cmd_eval,
    "serve": cmd_serve,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)

    command = sys.argv[1]
    remaining = sys.argv[2:]

    if command == "serve":
        cmd_serve()
    else:
        COMMANDS[command](remaining)
