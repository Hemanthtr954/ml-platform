"""
RAG Chain — wires retriever + prompt + LLM into a single callable.

LLM backends (controlled by LLMBackend enum):
  OLLAMA       — local model via Ollama (mistral, llama3, phi3, gemma2)
  HUGGINGFACE  — HuggingFace Hub inference API (free tier available)
  HF_LOCAL     — HuggingFace Transformers pipeline running on-device
  OPENAI       — OpenAI-compatible endpoint (bring-your-own key)

The chain uses LangChain Expression Language (LCEL):
  { context, question } | prompt | llm | StrOutputParser

For multi-turn chat, swap StrOutputParser for ConversationBufferWindowMemory.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from retrieval.retriever import RAGRetriever, RetrievalResult

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Answer the question using ONLY the context provided below.
If the answer is not found in the context, respond with: "I don't have enough information in the provided documents to answer this."
Do not make up information or use knowledge outside the context.

Context:
{context}

Question: {question}

Answer:"""


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    HF_LOCAL = "hf_local"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    backend: LLMBackend = LLMBackend.OLLAMA
    model_name: str = "mistral"             # ollama: mistral/llama3/phi3 | hf: mistralai/Mistral-7B-Instruct-v0.2
    temperature: float = 0.1
    max_tokens: int = 512
    ollama_base_url: str = "http://localhost:11434"
    hf_api_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # override for local OpenAI-compatible servers


@dataclass
class RAGResponse:
    answer: str
    query: str
    retrieval: RetrievalResult
    llm_backend: str
    model_name: str
    total_latency_ms: float
    sources: list[dict] = field(default_factory=list)


class RAGChain:
    """
    End-to-end RAG chain: retrieve → format context → prompt → LLM → answer.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_config: Optional[LLMConfig] = None,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ):
        self.retriever = retriever
        self.llm_config = llm_config or LLMConfig()
        self.prompt_template = prompt_template
        self._chain = None

    def _build_llm(self):
        cfg = self.llm_config

        if cfg.backend == LLMBackend.OLLAMA:
            from langchain_community.llms import Ollama
            return Ollama(
                model=cfg.model_name,
                temperature=cfg.temperature,
                base_url=cfg.ollama_base_url,
                num_predict=cfg.max_tokens,
            )

        if cfg.backend == LLMBackend.HUGGINGFACE:
            from langchain_community.llms import HuggingFaceHub
            token = cfg.hf_api_token or os.getenv("HF_API_TOKEN")
            return HuggingFaceHub(
                repo_id=cfg.model_name,
                huggingfacehub_api_token=token,
                model_kwargs={"temperature": cfg.temperature, "max_new_tokens": cfg.max_tokens},
            )

        if cfg.backend == LLMBackend.HF_LOCAL:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                do_sample=cfg.temperature > 0,
            )
            return HuggingFacePipeline(pipeline=pipe)

        if cfg.backend == LLMBackend.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cfg.model_name,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                api_key=cfg.openai_api_key or os.getenv("OPENAI_API_KEY"),
                base_url=cfg.openai_base_url,
            )

        raise ValueError(f"Unknown LLM backend: {cfg.backend}")

    def _build_chain(self):
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        llm = self._build_llm()
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain, llm

    def query(self, question: str) -> RAGResponse:
        t0 = time.time()

        # Retrieve relevant chunks
        retrieval = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieval)

        # Build chain on first call (lazy init — model loading is slow)
        if self._chain is None:
            logger.info(f"[RAGChain] Initializing LLM backend: {self.llm_config.backend}")
            self._chain, _ = self._build_chain()

        # Run the chain
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = self._build_llm()
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": question})

        total_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"[RAGChain] Query answered in {total_ms}ms "
            f"({retrieval.total_retrieved} chunks, {self.llm_config.backend})"
        )

        sources = [
            {
                "source": c.source,
                "page": c.page,
                "excerpt": c.content[:200],
            }
            for c in retrieval.chunks
        ]

        return RAGResponse(
            answer=answer.strip(),
            query=question,
            retrieval=retrieval,
            llm_backend=self.llm_config.backend.value,
            model_name=self.llm_config.model_name,
            total_latency_ms=total_ms,
            sources=sources,
        )

    def stream(self, question: str):
        """Yield answer tokens as they are generated (streaming mode)."""
        retrieval = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieval)

        llm = self._build_llm()
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | llm | StrOutputParser()

        for token in chain.stream({"context": context, "question": question}):
            yield token
