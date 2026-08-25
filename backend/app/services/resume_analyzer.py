import re
from typing import Any


def extract_email(text: str) -> str | None:
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    match = re.search(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )

    return match.group(0) if match else None


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    return 0.0


def extract_skills(text: str) -> list[str]:
    known_skills = [
        "Python",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "Power BI",
        "Tableau",
        "Excel",
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Data Science",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "FastAPI",
        "Flask",
        "Django",
        "React",
        "JavaScript",
        "HTML",
        "CSS",
        "Node.js",
        "MongoDB",
        "Git",
        "GitHub"
    ]

    text_lower = text.lower()

    found_skills = []

    for skill in known_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    return found_skills


def extract_name(text: str) -> str | None:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    # Usually the candidate name appears near the beginning.
    for line in lines[:10]:

        if (
            "resume" not in line.lower()
            and "curriculum vitae" not in line.lower()
            and "cv" != line.lower()
            and len(line.split()) <= 5
            and not re.search(r"[@|:/\\]", line)
        ):
            return line

    return None


def analyze_resume(text: str) -> dict[str, Any]:
    return {
        "candidate_name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "text_length": len(text)
    }