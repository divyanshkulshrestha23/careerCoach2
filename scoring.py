"""Local job-fit score: how many of the skills a job asks for appear in a resume.

No LLM or network calls are involved, so the same inputs always give the same score.
The score is only as good as SKILLS: a skill missing from the list is never counted.
"""

import re

# Canonical skill names. Phrases are matched as whole words, so "rest api" will not match "restaurant".
SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "golang", "c++", "c#", "kotlin", "swift", "rust", "scala",
    "sql", "bash", "html", "css",
    # Frameworks and libraries
    "react", "next.js", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot", "express",
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "keras", "langchain", "spark",
    "pyspark", "hadoop", "kafka", "airflow", "dbt", "gradio", "streamlit", "langgraph", "vllm",
    # Data and storage
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "chromadb", "snowflake",
    "bigquery", "data warehouse", "etl", "data pipeline",
    # Cloud and infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "linux",
    "ci/cd", "github actions", "jenkins", "serverless", "cloud run", "lambda", "dynamodb", "sagemaker", "vertex ai",
    # Practices and concepts
    "rest api", "graphql", "grpc", "oauth", "jwt", "microservices", "git", "unit testing",
    "test driven development", "grafana", "opentelemetry", "temporal",
    "agile", "scrum", "system design", "data structures", "algorithms", "distributed systems",
    "machine learning", "deep learning", "natural language processing", "computer vision",
    "statistics", "data analysis", "data visualization", "a/b testing", "feature engineering",
    "model deployment", "mlops", "llm", "retrieval augmented generation", "prompt engineering",
    "ocr", "object detection",
]

# Common alternative spellings mapped to a canonical skill.
ALIASES = {
    "ts": "typescript",
    "nodejs": "node.js",
    "node": "node.js",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "google cloud": "gcp",
    "amazon web services": "aws",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "rag": "retrieval augmented generation",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "sklearn": "scikit-learn",
}

_PATTERN_CACHE: list[tuple[re.Pattern, str]] = []


def _build_patterns() -> list[tuple[re.Pattern, str]]:
    if not _PATTERN_CACHE:
        terms = {s: s for s in SKILLS} | ALIASES
        for term, canonical in terms.items():
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")
            _PATTERN_CACHE.append((pattern, canonical))
    return _PATTERN_CACHE


def extract_skills(text: str) -> set[str]:
    """Return the canonical skills mentioned in a piece of text."""
    normalized = " ".join(text.lower().split())
    found = set()
    for pattern, canonical in _build_patterns():
        if pattern.search(normalized):
            found.add(canonical)
    return found


def match_resume(resume_text: str, job_text: str) -> dict:
    """Compare a resume with a job description.

    Returns the score (0-100, or None when the job lists no recognised skills),
    the skills the job asks for that the resume has, and the ones it does not.
    """
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)

    if not job_skills:
        return {"score": None, "job_skills": [], "matched": [], "missing": []}

    matched = sorted(job_skills & resume_skills)
    missing = sorted(job_skills - resume_skills)
    score = round(100 * len(matched) / len(job_skills))
    return {"score": score, "job_skills": sorted(job_skills), "matched": matched, "missing": missing}


def format_report(result: dict) -> str:
    """Plain-text report for the app."""
    if result["score"] is None:
        return ("No recognised skills were found in the job description, so no score was computed. "
                "Paste a fuller job description, or check that the listed skills are in SKILLS in scoring.py.")

    lines = [
        f"Match score: {result['score']}/100",
        f"The job asks for {len(result['job_skills'])} recognised skills; your resume shows {len(result['matched'])} of them.",
        "",
        "Found in your resume: " + (", ".join(result["matched"]) or "none"),
        "Missing from your resume: " + (", ".join(result["missing"]) or "none"),
        "",
        "This score only counts skills from the built-in list. It does not measure experience depth or seniority.",
    ]
    return "\n".join(lines)
