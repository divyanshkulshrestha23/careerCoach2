"""Small pure helpers used by processor.py. Kept free of heavy imports so they can be tested alone."""

# Text returned by filestore.query_relevant_experience when no resume is stored yet.
NO_RESUME_MARKER = "No resume data found"


def choose_context(retrieved: str, fallback: str) -> str:
    """Use retrieved resume excerpts when present, otherwise the manually pasted resume text."""
    if retrieved and NO_RESUME_MARKER not in retrieved:
        return retrieved
    return fallback


def response_text(content) -> str:
    """Return plain text from an LLM message's content, which may be a string or a list of parts."""
    if isinstance(content, list):
        if not content:
            return ""
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return str(first)
    return content
