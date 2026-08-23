from typing import Any

from app.services.semantic_matcher import (
    calculate_semantic_similarity
)


# ============================================================
# Skill normalization
# ============================================================

def normalize_skill(skill: str) -> str:

    if not skill:
        return ""

    skill = str(skill).lower().strip()

    replacements = {
        "-": " ",
        "_": " ",
        ".": "",
        "/": " ",
    }

    for old, new in replacements.items():
        skill = skill.replace(old, new)

    skill = " ".join(skill.split())

    aliases = {
        "powerbi": "power bi",
        "ms excel": "excel",
        "microsoft excel": "excel",
        "sklearn": "scikit learn",
        "scikit-learn": "scikit learn",
        "machinelearning": "machine learning",
        "artificial intelligence": "ai",
    }

    return aliases.get(skill, skill)


# ============================================================
# Normalize skill list
# ============================================================

def normalize_skill_list(
    skills: list[str] | None
) -> list[str]:

    if not skills:
        return []

    normalized = []

    for skill in skills:

        value = normalize_skill(skill)

        if value:
            normalized.append(value)

    return sorted(set(normalized))


# ============================================================
# Required skill matching
# ============================================================

def calculate_skill_match(
    required_skills: list[str],
    candidate_skills: list[str]
) -> dict[str, Any]:

    required = set(
        normalize_skill_list(required_skills)
    )

    candidate = set(
        normalize_skill_list(candidate_skills)
    )

    if not required:

        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": []
        }

    matched = required.intersection(candidate)

    missing = required.difference(candidate)

    score = (
        len(matched) /
        len(required)
    ) * 100

    return {
        "score": round(score, 2),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }


# ============================================================
# Preferred skill matching
# ============================================================

def calculate_preferred_skill_match(
    preferred_skills: list[str],
    candidate_skills: list[str]
) -> dict[str, Any]:

    preferred = set(
        normalize_skill_list(preferred_skills)
    )

    candidate = set(
        normalize_skill_list(candidate_skills)
    )

    if not preferred:

        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": []
        }

    matched = preferred.intersection(candidate)

    missing = preferred.difference(candidate)

    score = (
        len(matched) /
        len(preferred)
    ) * 100

    return {
        "score": round(score, 2),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }


# ============================================================
# Experience score
# ============================================================

def calculate_experience_score(
    minimum_required: float,
    candidate_experience: float
) -> float:

    minimum_required = float(
        minimum_required or 0
    )

    candidate_experience = float(
        candidate_experience or 0
    )

    if minimum_required <= 0:
        return 100.0

    if candidate_experience >= minimum_required:
        return 100.0

    if candidate_experience <= 0:
        return 0.0

    score = (
        candidate_experience /
        minimum_required
    ) * 100

    return round(
        min(score, 100.0),
        2
    )


# ============================================================
# Final score
# ============================================================

def calculate_final_score(
    skill_score: float,
    preferred_skill_score: float,
    experience_score: float,
    semantic_score: float
) -> float:

    final_score = (
        skill_score * 0.40
        + preferred_skill_score * 0.15
        + experience_score * 0.20
        + semantic_score * 0.25
    )

    return round(
        final_score,
        2
    )


# ============================================================
# Recommendation
# ============================================================

def generate_recommendation(
    final_score: float
) -> str:

    if final_score >= 80:
        return "Strong Match"

    if final_score >= 65:
        return "Good Match"

    if final_score >= 50:
        return "Moderate Match"

    return "Low Match"


# ============================================================
# Main matching engine
# ============================================================

def match_candidate(
    job: dict,
    candidate: dict,
    resume_text: str = ""
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Job information
    # --------------------------------------------------------

    required_skills = job.get(
        "required_skills",
        []
    )

    preferred_skills = job.get(
        "preferred_skills",
        []
    )

    minimum_experience = job.get(
        "minimum_experience_years",
        0
    )

    job_text = job.get(
        "job_description",
        ""
    )

    # --------------------------------------------------------
    # Candidate information
    # --------------------------------------------------------

    candidate_skills = candidate.get(
        "skills",
        []
    )

    candidate_experience = candidate.get(
        "experience_years",
        0
    )

    # --------------------------------------------------------
    # Required skills
    # --------------------------------------------------------

    required_result = calculate_skill_match(
        required_skills,
        candidate_skills
    )

    # --------------------------------------------------------
    # Preferred skills
    # --------------------------------------------------------

    preferred_result = calculate_preferred_skill_match(
        preferred_skills,
        candidate_skills
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    experience_score = calculate_experience_score(
        minimum_required=minimum_experience,
        candidate_experience=candidate_experience
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_score = calculate_semantic_similarity(
        job_text,
        resume_text,
        candidate_skills
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    final_score = calculate_final_score(
        skill_score=required_result["score"],
        preferred_skill_score=preferred_result["score"],
        experience_score=experience_score,
        semantic_score=semantic_score
    )

    # --------------------------------------------------------
    # Return ONLY matching information
    #
    # SHAP is intentionally NOT calculated here.
    # --------------------------------------------------------

    return {

        "overall_score": final_score,

        "skill_score": (
            required_result["score"]
        ),

        "preferred_skill_score": (
            preferred_result["score"]
        ),

        "experience_score": (
            experience_score
        ),

        "semantic_score": (
            semantic_score
        ),

        "matched_required_skills": (
            required_result["matched_skills"]
        ),

        "missing_required_skills": (
            required_result["missing_skills"]
        ),

        "matched_preferred_skills": (
            preferred_result["matched_skills"]
        ),

        "missing_preferred_skills": (
            preferred_result["missing_skills"]
        ),

        "recommendation": (
            generate_recommendation(
                final_score
            )
        )
    }