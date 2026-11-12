"""Tests for the local skill-match score."""

from scoring import extract_skills, format_report, match_resume


def test_skills_are_found_case_insensitively():
    assert {"python", "docker"} <= extract_skills("Built services in PYTHON and Docker")


def test_whole_word_matching_avoids_substring_hits():
    # "rest api" must not match "restaurant", and "java" must not match "javascript".
    skills = extract_skills("Managed a restaurant team and wrote javascript")
    assert "rest api" not in skills
    assert "java" not in skills
    assert "javascript" in skills


def test_aliases_map_to_canonical_skill():
    assert "kubernetes" in extract_skills("Ran workloads on k8s")
    assert "postgresql" in extract_skills("Tuned Postgres queries")
    assert "retrieval augmented generation" in extract_skills("Built a RAG system")


def test_score_is_share_of_job_skills_found():
    result = match_resume(
        resume_text="Python, Docker and SQL",
        job_text="Requires Python, Docker, Kubernetes and AWS",
    )
    assert result["score"] == 50
    assert result["matched"] == ["docker", "python"]
    assert result["missing"] == ["aws", "kubernetes"]


def test_full_match_scores_100():
    result = match_resume("Python and SQL", "Need Python and SQL")
    assert result["score"] == 100
    assert result["missing"] == []


def test_no_recognised_skills_gives_no_score():
    result = match_resume("Python developer", "Looking for a barista")
    assert result["score"] is None
    assert "no score was computed" in format_report(result)


def test_skill_match_is_case_insensitive_on_resume_side():
    result = match_resume("Built APIs in PYTHON", "Need python")
    assert result["score"] == 100


def test_node_js_does_not_count_as_javascript():
    skills = extract_skills("Lambda functions in Node.js and Python")
    assert "node.js" in skills
    assert "javascript" not in skills


def test_next_js_is_recognised():
    assert "next.js" in extract_skills("Built the UI in Next.js")


def test_common_word_go_is_not_golang():
    assert "golang" not in extract_skills("We go to market in Q3")
