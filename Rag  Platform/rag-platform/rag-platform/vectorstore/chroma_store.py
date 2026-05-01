"""
ChromaDB Vector Store — persistent local vector database.

Embedding models (in order of preference):
  1. sentence-transformers/all-MiniLM-L6-v2  — fast, 80 MB, CPU-friendly
  2. sentence-transformers/all-mpnet-base-v2  — slower but higher quality
  3. BAAI/bge-small-en-v1.5                   — best quality/size ratio

ChromaDB persists to disk automatically; no server needed for local use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./vectorstore_data"
DEFAULT_COLLECTION = "rag_documents"


class EmbeddingModel(str, Enum):
    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET = "sentence-transformers/all-mpnet-base-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"


@dataclass
class VectorStoreConfig:
    persist_directory: str = DEFAULT_PERSIST_DIR
    collection_name: str = DEFAULT_COLLECTION
    embedding_model: EmbeddingModel = EmbeddingModel.MINI_LM
    device: str = "cpu"


@dataclass
class StoreStats:
    collection_name: str
    total_documents: int
    persist_directory: str
    embedding_model: str


class ChromaVectorStore:
    """
    Thin wrapper around LangChain's Chroma integration.
    Handles creation, persistence, and incremental updates.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._db = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model.value,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"[VectorStore] Loaded embeddings: {self.config.embedding_model.value}")
        return self._embeddings

    def _get_db(self):
        if self._db is None:
            from langchain_chroma import Chroma
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            self._db = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self._get_embeddings(),
                persist_directory=self.config.persist_directory,
            )
            logger.info(
                f"[VectorStore] Opened collection '{self.config.collection_name}' "
                f"at {self.config.persist_directory}"
            )
        return self._db

    def add_documents(self, chunks: list) -> int:
        db = self._get_db()
        db.add_documents(chunks)
        count = db._collection.count()
        logger.info(f"[VectorStore] Added chunks. Total documents: {count}")
        return count

    def build_from_chunks(self, chunks: list) -> int:
        """Drop existing collection and rebuild from scratch."""
        from langchain_chroma import Chroma
        import shutil

        persist_path = Path(self.config.persist_directory)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            logger.info(f"[VectorStore] Cleared existing store at {persist_path}")

        self._db = None  # force re-init
        persist_path.mkdir(parents=True, exist_ok=True)

        self._db = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
        )
        count = self._db._collection.count()
        logger.info(f"[VectorStore] Built store with {count} chunks")
        return count

    def as_retriever(
        self,
        k: int = 4,
        search_type: str = "mmr",
        fetch_k: int = 20,
        filter: Optional[dict] = None,
    ):
        db = self._get_db()
        search_kwargs: dict = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
        if filter:
            search_kwargs["filter"] = filter
        return db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 4) -> list:
        db = self._get_db()
        return db.similarity_search(query, k=k)

    def stats(self) -> StoreStats:
        db = self._get_db()
        count = db._collection.count()
        return StoreStats(
            collection_name=self.config.collection_name,
            total_documents=count,
            persist_directory=self.config.persist_directory,
            embedding_model=self.config.embedding_model.value,
        )
