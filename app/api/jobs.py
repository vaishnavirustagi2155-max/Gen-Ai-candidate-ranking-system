from fastapi import APIRouter
from pydantic import BaseModel, Field
import re

from app.services.matching_engine import match_candidate


router = APIRouter(
    prefix="/api/jobs",
    tags=["Jobs"]
)


# ============================================================
# Request Models
# ============================================================

class JobDescriptionRequest(BaseModel):

    job_description: str = Field(
        ...,
        min_length=20,
        description="Complete job description"
    )


class CandidateProfile(BaseModel):

    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    experience_years: float = 0.0

    text_length: int = 0


class CandidateMatchRequest(BaseModel):

    job: dict

    candidate: CandidateProfile

    resume_text: str = Field(
        ...,
        min_length=20
    )


# ============================================================
# Skill Dictionary
# ============================================================

SKILL_ALIASES = {

    "python": "python",
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",

    "power bi": "power bi",
    "powerbi": "power bi",

    "tableau": "tableau",

    "excel": "excel",
    "microsoft excel": "excel",
    "ms excel": "excel",

    "machine learning": "machine learning",
    "machine-learning": "machine learning",
    "ml": "machine learning",

    "deep learning": "deep learning",
    "deep-learning": "deep learning",
    "dl": "deep learning",

    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",

    "pandas": "pandas",
    "numpy": "numpy",

    "scikit-learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",

    "java": "java",
    "javascript": "javascript",
    "typescript": "typescript",

    "react": "react",
    "node.js": "node.js",
    "nodejs": "node.js",

    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",

    "docker": "docker",
    "kubernetes": "kubernetes",

    "git": "git",
    "github": "github",

    "spark": "spark",
    "hadoop": "hadoop",

    "flask": "flask",
    "fastapi": "fastapi",

    "mongodb": "mongodb",
}


# ============================================================
# Extract Skills
# ============================================================

def extract_skills(text: str) -> list[str]:

    text_lower = text.lower()

    found_skills = []

    for skill_name, normalized_name in SKILL_ALIASES.items():

        # Escape special characters in skill names
        pattern = r"(?<!\w)" + re.escape(skill_name) + r"(?!\w)"

        if re.search(pattern, text_lower):

            if normalized_name not in found_skills:

                found_skills.append(
                    normalized_name
                )

    return found_skills


# ============================================================
# Extract Experience Requirement
# ============================================================

def extract_minimum_experience(
    text: str
) -> float:

    text_lower = text.lower()

    patterns = [

        # 1-2 years of experience
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*years?",

        # minimum 2 years
        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*years?",

        # at least 2 years
        r"at\s+least\s+(\d+(?:\.\d+)?)\s*years?",

        # 2+ years
        r"(\d+(?:\.\d+)?)\s*\+\s*years?",

        # 2 years of experience
        r"(\d+(?:\.\d+)?)\s*years?\s+of\s+experience",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except (ValueError, IndexError):
                pass

    return 0.0


# ============================================================
# Determine Required vs Preferred Skills
# ============================================================

def extract_job_requirements(
    job_description: str
) -> tuple[list[str], list[str]]:

    text_lower = job_description.lower()

    all_skills = extract_skills(
        job_description
    )

    required_skills = []
    preferred_skills = []

    # --------------------------------------------------------
    # Preferred section detection
    # --------------------------------------------------------

    preferred_keywords = [
        "preferred",
        "prefer",
        "nice to have",
        "good to have",
        "bonus",
        "plus",
        "desired"
    ]

    # --------------------------------------------------------
    # Find skills appearing near preferred wording
    # --------------------------------------------------------

    for skill in all_skills:

        skill_pattern = re.escape(skill)

        skill_position = text_lower.find(
            skill.lower()
        )

        if skill_position == -1:
            continue

        # Look at the sentence containing the skill
        sentence_start = text_lower.rfind(
            ".",
            0,
            skill_position
        )

        sentence_end = text_lower.find(
            ".",
            skill_position
        )

        if sentence_start == -1:
            sentence_start = 0
        else:
            sentence_start += 1

        if sentence_end == -1:
            sentence_end = len(text_lower)

        sentence = text_lower[
            sentence_start:sentence_end
        ]

        is_preferred = any(
            keyword in sentence
            for keyword in preferred_keywords
        )

        if is_preferred:

            if skill not in preferred_skills:
                preferred_skills.append(skill)

        else:

            if skill not in required_skills:
                required_skills.append(skill)

    # --------------------------------------------------------
    # Fallback:
    # If no required skills were identified but skills exist,
    # treat non-preferred skills as required.
    # --------------------------------------------------------

    for skill in all_skills:

        if (
            skill not in preferred_skills
            and skill not in required_skills
        ):
            required_skills.append(skill)

    return (
        sorted(required_skills),
        sorted(preferred_skills)
    )


# ============================================================
# Analyze Job
# ============================================================

@router.post("/analyze")
async def analyze_job(
    request: JobDescriptionRequest
):

    job_description = (
        request.job_description.strip()
    )

    # --------------------------------------------------------
    # Extract experience
    # --------------------------------------------------------

    minimum_experience_years = (
        extract_minimum_experience(
            job_description
        )
    )

    # --------------------------------------------------------
    # Extract required/preferred skills
    # --------------------------------------------------------

    (
        required_skills,
        preferred_skills
    ) = extract_job_requirements(
        job_description
    )

    # --------------------------------------------------------
    # Structured job object
    # --------------------------------------------------------

    job = {

        "job_description": job_description,

        "required_skills": required_skills,

        "preferred_skills": preferred_skills,

        "minimum_experience_years": (
            minimum_experience_years
        )
    }

    return {

        "message": "Job analyzed successfully",

        "character_count": len(
            job_description
        ),

        "job_description": job_description,

        "required_skills": required_skills,

        "preferred_skills": preferred_skills,

        "minimum_experience_years": (
            minimum_experience_years
        ),

        "job": job
    }


# ============================================================
# Match Single Candidate
# ============================================================

@router.post("/match")
async def match_candidate_to_job(
    request: CandidateMatchRequest
):

    result = match_candidate(

        job=request.job,

        candidate=(
            request.candidate.model_dump()
        ),

        resume_text=request.resume_text
    )

    return {

        "message": "Candidate matching completed",

        "result": result
    }