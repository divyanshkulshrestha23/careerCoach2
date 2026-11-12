"""Tests for the context and response helpers used by the LLM prompts."""

from helpers import NO_RESUME_MARKER, choose_context, response_text


def test_retrieved_excerpts_are_preferred_over_fallback():
    assert choose_context("Built a billing service", "pasted text") == "Built a billing service"


def test_fallback_used_when_nothing_retrieved():
    assert choose_context("", "pasted text") == "pasted text"


def test_fallback_used_when_database_is_empty():
    retrieved = f"{NO_RESUME_MARKER} in the database. Please upload and parse a resume first."
    assert choose_context(retrieved, "pasted text") == "pasted text"


def test_response_text_from_plain_string():
    assert response_text("hello") == "hello"


def test_response_text_from_list_of_parts():
    assert response_text([{"type": "text", "text": "hello"}]) == "hello"


def test_response_text_from_empty_list():
    assert response_text([]) == ""


def test_response_text_from_list_of_strings():
    assert response_text(["hello"]) == "hello"
