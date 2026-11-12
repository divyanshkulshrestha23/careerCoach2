"""Tests for the semantic coverage signal, using a stand-in embedder so no model download is needed."""

from semantic import coverage, split_sentences


# Fixed vocabulary so every call returns vectors of the same length, as a real embedding model does.
_VOCAB = sorted({w for t in [
    "Build retrieval pipelines over internal documents for question answering.",
    "Design distributed databases for financial transaction settlement systems.",
    "Painted murals and organised community gardens on weekends with volunteers.",
    "Design distributed databases for financial settlement.",
    "Lead a team of volunteers organising community gardens.",
    "Built retrieval pipelines over documents.",
    "Organised volunteers for community gardens.",
    "Short.", "This sentence is long enough to keep here.", "Another one that is long enough.",
    "Some job text that is long enough to count.",
] for w in t.lower().replace(".", "").split()})


def bag_of_words_embedder(texts):
    """Each text becomes a vector of word counts over a fixed vocabulary."""
    return [[float(t.lower().replace(".", "").split().count(w)) for w in _VOCAB] for t in texts]


def test_split_drops_short_fragments():
    sentences = split_sentences("Short. This sentence is long enough to keep here. Another one that is long enough.")
    assert all(len(s) >= 25 for s in sentences)
    assert len(sentences) == 2


def test_identical_text_has_full_coverage():
    text = "Build retrieval pipelines over internal documents for question answering."
    result = coverage(text, text, bag_of_words_embedder)
    assert result["coverage"] == 1.0


def test_unrelated_text_has_low_coverage():
    job = "Design distributed databases for financial transaction settlement systems."
    resume = "Painted murals and organised community gardens on weekends with volunteers."
    result = coverage(resume, job, bag_of_words_embedder)
    assert result["coverage"] < 0.3


def test_weakest_sentences_are_reported():
    job = ("Design distributed databases for financial settlement. "
           "Lead a team of volunteers organising community gardens.")
    resume = "Built retrieval pipelines over documents. Organised volunteers for community gardens."
    result = coverage(resume, job, bag_of_words_embedder)
    assert result["weakest"][0]["job_sentence"].startswith("Design distributed databases")


def test_empty_inputs_return_no_coverage():
    assert coverage("", "Some job text that is long enough to count.", bag_of_words_embedder)["coverage"] is None
