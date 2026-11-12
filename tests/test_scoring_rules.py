"""Tests for the extraction fixes and the minimum-skills rule."""

from scoring import MIN_JOB_SKILLS, extract_skills, match_resume


def test_gemini_counts_as_llm():
    assert "llm" in extract_skills("Built a service on the Gemini API")


def test_langchain_and_protobufs_are_recognised():
    skills = extract_skills("LangChain pipelines; strongly typed protobufs")
    assert "langchain" in skills
    assert "protobuf" in skills


def test_short_job_list_gets_no_score():
    result = match_resume("Python and SQL", "Need Python, SQL, Docker and AWS")
    assert len(result["job_skills"]) < MIN_JOB_SKILLS
    assert result["score"] is None
    assert result["reason"] == "too_few_skills"


def test_job_with_enough_skills_gets_a_score():
    result = match_resume("Python, SQL, Docker, AWS and Kubernetes",
                          "Need Python, SQL, Docker, AWS and Kubernetes")
    assert result["score"] == 100
