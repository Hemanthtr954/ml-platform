"""
RAG Evaluation — scores retrieval quality and answer faithfulness.

Metrics (no external LLM required for the first three):
  context_recall    — fraction of retrieved chunks that are relevant (keyword overlap)
  answer_relevance  — answer mentions key terms from the question
  faithfulness      — answer tokens are grounded in the retrieved context
  ragas_score       — optional: full RAGAS suite (requires pip install ragas)

These lightweight heuristics run locally without any API calls.
For production, swap in the RAGAS library for LLM-graded scores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    retrieved_contexts: list[str]
    generated_answer: str


@dataclass
class EvalScore:
    context_recall: float       # 0–1: how many retrieved chunks contain relevant terms
    answer_relevance: float     # 0–1: answer addresses the question
    faithfulness: float         # 0–1: answer tokens grounded in context
    overall: float              # weighted average
    details: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    total_cases: int
    avg_context_recall: float
    avg_answer_relevance: float
    avg_faithfulness: float
    avg_overall: float
    scores: list[EvalScore] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    pass_threshold: float = 0.6

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer for overlap scoring."""
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "or", "it"}
    return {t for t in tokens if t not in stopwords and len(t) > 2}


def score_context_recall(question: str, contexts: list[str]) -> float:
    """Fraction of contexts that share key terms with the question."""
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    relevant = sum(
        1 for ctx in contexts
        if len(q_tokens & _tokenize(ctx)) / len(q_tokens) >= 0.2
    )
    return relevant / len(contexts)


def score_answer_relevance(question: str, answer: str) -> float:
    """How many key question terms appear in the answer."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    a_tokens = _tokenize(answer)
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    return min(overlap * 1.5, 1.0)  # scale up since answers are longer


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentence-level claims that appear verbatim in context."""
    if not contexts or not answer:
        return 0.0
    full_context = " ".join(contexts).lower()
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded = 0
    for sentence in sentences:
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        # A sentence is grounded if ≥40% of its non-trivial tokens appear in context
        found = sum(1 for t in tokens if t in full_context)
        if found / len(tokens) >= 0.4:
            grounded += 1

    return grounded / len(sentences)


class RAGEvaluator:
    """
    Evaluates a set of (question, context, answer) triples.
    Works entirely offline — no API calls needed.
    """

    WEIGHTS = {"context_recall": 0.3, "answer_relevance": 0.3, "faithfulness": 0.4}

    def __init__(self, pass_threshold: float = 0.6):
        self.pass_threshold = pass_threshold

    def score_case(self, case: EvalCase) -> EvalScore:
        cr = score_context_recall(case.question, case.retrieved_contexts)
        ar = score_answer_relevance(case.question, case.generated_answer)
        fa = score_faithfulness(case.generated_answer, case.retrieved_contexts)

        overall = (
            cr * self.WEIGHTS["context_recall"]
            + ar * self.WEIGHTS["answer_relevance"]
            + fa * self.WEIGHTS["faithfulness"]
        )

        return EvalScore(
            context_recall=round(cr, 3),
            answer_relevance=round(ar, 3),
            faithfulness=round(fa, 3),
            overall=round(overall, 3),
            details={"expected": case.expected_answer[:100]},
        )

    def evaluate(self, cases: list[EvalCase]) -> EvalReport:
        scores = [self.score_case(c) for c in cases]

        passed = sum(1 for s in scores if s.overall >= self.pass_threshold)
        n = len(scores)

        report = EvalReport(
            total_cases=n,
            avg_context_recall=round(sum(s.context_recall for s in scores) / n, 3),
            avg_answer_relevance=round(sum(s.answer_relevance for s in scores) / n, 3),
            avg_faithfulness=round(sum(s.faithfulness for s in scores) / n, 3),
            avg_overall=round(sum(s.overall for s in scores) / n, 3),
            scores=scores,
            passed=passed,
            failed=n - passed,
            pass_threshold=self.pass_threshold,
        )

        logger.info(
            f"[Eval] {n} cases | overall={report.avg_overall:.3f} "
            f"| pass_rate={report.pass_rate:.1%} ({passed}/{n})"
        )
        return report

    def evaluate_from_rag_chain(self, chain, qa_pairs: list[dict]) -> EvalReport:
        """
        Convenience method: run the RAG chain on each question, then evaluate.
        qa_pairs: [{"question": ..., "expected_answer": ...}, ...]
        """
        cases = []
        for pair in qa_pairs:
            response = chain.query(pair["question"])
            cases.append(EvalCase(
                question=pair["question"],
                expected_answer=pair.get("expected_answer", ""),
                retrieved_contexts=[c.content for c in response.retrieval.chunks],
                generated_answer=response.answer,
            ))
        return self.evaluate(cases)
