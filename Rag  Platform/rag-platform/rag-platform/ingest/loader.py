"""
Document Ingest Pipeline — load, split, and index source documents.

Supported formats:
  .pdf   — via PyPDFLoader (page-level metadata)
  .txt   — via TextLoader
  .md    — via TextLoader
  .csv   — via CSVLoader
  URL    — via WebBaseLoader (scrape and chunk a web page)

Chunking strategy:
  RecursiveCharacterTextSplitter with configurable chunk_size / overlap.
  Smaller chunks (256–512) → precise retrieval.
  Larger chunks (1024–2048) → richer context per chunk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " "])


@dataclass
class IngestResult:
    total_documents: int
    total_chunks: int
    sources: list[str]
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)


class DocumentIngestor:
    """
    Loads documents from a directory or individual file paths, splits them
    into chunks, and returns LangChain Document objects ready for embedding.
    """

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()

    def load_directory(self, directory: str) -> list:
        from langchain_community.document_loaders import (
            DirectoryLoader,
            PyPDFLoader,
            TextLoader,
            CSVLoader,
        )

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        docs = []
        errors = []

        loaders = [
            ("**/*.pdf", PyPDFLoader),
            ("**/*.txt", TextLoader),
            ("**/*.md", TextLoader),
        ]

        for glob_pattern, loader_cls in loaders:
            try:
                loader = DirectoryLoader(
                    str(path),
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    silent_errors=True,
                )
                loaded = loader.load()
                docs.extend(loaded)
                logger.info(f"[Ingest] {glob_pattern}: {len(loaded)} docs")
            except Exception as e:
                errors.append(f"{glob_pattern}: {e}")
                logger.warning(f"[Ingest] Failed to load {glob_pattern}: {e}")

        # CSV handled separately (needs different kwargs)
        for csv_file in path.rglob("*.csv"):
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                docs.extend(loaded)
                logger.info(f"[Ingest] CSV {csv_file.name}: {len(loaded)} rows")
            except Exception as e:
                errors.append(f"{csv_file}: {e}")

        return docs, errors

    def load_urls(self, urls: list[str]) -> tuple[list, list[str]]:
        from langchain_community.document_loaders import WebBaseLoader

        docs = []
        errors = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                loaded = loader.load()
                docs.extend(loaded)
                logger.info(f"[Ingest] URL {url}: {len(loaded)} docs")
            except Exception as e:
                errors.append(f"{url}: {e}")
                logger.warning(f"[Ingest] Failed to load URL {url}: {e}")
        return docs, errors

    def split(self, docs: list) -> list:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"[Ingest] Split {len(docs)} docs → {len(chunks)} chunks")
        return chunks

    def ingest_directory(self, directory: str) -> tuple[list, IngestResult]:
        t0 = time.time()
        docs, errors = self.load_directory(directory)
        chunks = self.split(docs)
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        result = IngestResult(
            total_documents=len(docs),
            total_chunks=len(chunks),
            sources=sources,
            elapsed_seconds=round(time.time() - t0, 2),
            errors=errors,
        )
        return chunks, result

    def ingest_urls(self, urls: list[str]) -> tuple[list, IngestResult]:
        t0 = time.time()
        docs, errors = self.load_urls(urls)
        chunks = self.split(docs)
        result = IngestResult(
            total_documents=len(docs),
            total_chunks=len(chunks),
            sources=urls,
            elapsed_seconds=round(time.time() - t0, 2),
            errors=errors,
        )
        return chunks, result
