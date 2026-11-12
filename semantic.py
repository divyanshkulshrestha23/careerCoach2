"""Semantic coverage: for each sentence of a job description, how close is its best match in the resume?

This is a separate signal from the skill score. It can catch experience the skill list
doesn't name (for example, "built retrieval pipelines over documents" against a resume
that says "RAG pipelines"). It is not included in the skill score.
"""

import math
import re
from typing import Callable

MIN_SENTENCE_CHARS = 25


def split_sentences(text: str) -> list[str]:
    """Split text into sentences and drop fragments too short to say anything."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s*", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_CHARS]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def default_embedder() -> Callable[[list[str]], list[list[float]]]:
    """The same local embedding model the app uses for resume search (loaded on first use)."""
    from chromadb.utils import embedding_functions

    ef = embedding_functions.DefaultEmbeddingFunction()
    return lambda texts: [list(map(float, v)) for v in ef(texts)]


def coverage(resume_text: str, job_text: str, embed: Callable[[list[str]], list[list[float]]]) -> dict:
    """Return the average best-match similarity and the three job sentences the resume covers least."""
    job_sentences = split_sentences(job_text)
    resume_sentences = split_sentences(resume_text)
    if not job_sentences or not resume_sentences:
        return {"coverage": None, "weakest": []}

    job_vecs = embed(job_sentences)
    resume_vecs = embed(resume_sentences)

    best = []
    for sentence, jv in zip(job_sentences, job_vecs):
        sims = [_cosine(jv, rv) for rv in resume_vecs]
        best.append((sentence, max(sims)))

    average = sum(score for _, score in best) / len(best)
    weakest = sorted(best, key=lambda item: item[1])[:3]
    return {
        "coverage": round(average, 3),
        "weakest": [{"job_sentence": s, "best_similarity": round(v, 3)} for s, v in weakest],
    }
