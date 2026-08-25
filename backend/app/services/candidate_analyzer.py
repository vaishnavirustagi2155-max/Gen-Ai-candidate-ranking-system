import re


COMMON_SKILLS = [
    "python",
    "sql",
    "mysql",
    "postgresql",
    "power bi",
    "tableau",
    "excel",
    "machine learning",
    "deep learning",
    "data science",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "java",
    "javascript",
    "typescript",
    "react",
    "node.js",
    "mongodb",
    "git",
    "github",
    "aws",
    "azure",
    "docker",
]


def extract_email(text: str) -> str | None:

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:

    match = re.search(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        text
    )

    return match.group(0) if match else None


def extract_name(text: str) -> str | None:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    ignored_words = {
        "resume",
        "curriculum vitae",
        "cv",
        "profile",
    }

    for line in lines[:10]:

        lower_line = line.lower()

        if lower_line in ignored_words:
            continue

        if "@" in line:
            continue

        if any(char.isdigit() for char in line):
            continue

        words = line.split()

        if 2 <= len(words) <= 4:

            if all(
                word.replace(
                    "-", ""
                ).replace(
                    "'", ""
                ).isalpha()
                for word in words
            ):
                return line

    return None


def extract_skills(text: str) -> list[str]:

    text_lower = text.lower()

    found_skills = []

    for skill in COMMON_SKILLS:

        if skill.lower() in text_lower:
            found_skills.append(skill)

    return sorted(
        set(found_skills)
    )


def extract_experience_years(
    text: str
) -> float:

    text_lower = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*years?"
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text_lower
        )

        for value in matches:
            try:
                values.append(
                    float(value)
                )
            except ValueError:
                pass

    if not values:
        return 0.0

    return max(values)


def analyze_resume(
    resume_text: str
) -> dict:

    return {
        "candidate_name": extract_name(
            resume_text
        ),
        "email": extract_email(
            resume_text
        ),
        "phone": extract_phone(
            resume_text
        ),
        "skills": extract_skills(
            resume_text
        ),
        "experience_years": extract_experience_years(
            resume_text
        ),
        "text_length": len(
            resume_text
        )
    }