from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/ranking",
    tags=["Ranking"]
)


# ============================================================
# Candidate Profile
# ============================================================

class CandidateProfile(BaseModel):

    candidate_name: str | None = None

    email: str | None = None

    phone: str | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    experience_years: float = 0.0

    text_length: int = 0


# ============================================================
# Candidate for Ranking
# ============================================================

class CandidateForRanking(BaseModel):

    candidate: CandidateProfile

    resume_text: str = Field(
        ...,
        min_length=20
    )


# ============================================================
# Ranking Request
# ============================================================

class RankingRequest(BaseModel):

    job: dict

    candidates: list[CandidateForRanking] = Field(
        ...,
        min_length=1
    )


# ============================================================
# Rank Candidates
# ============================================================

@router.post("/rank")
async def rank_candidates(
    request: RankingRequest
):

    # ========================================================
    # IMPORTANT:
    # Heavy ML services are imported ONLY when ranking
    # is actually requested.
    #
    # This keeps FastAPI startup memory low on Render.
    # ========================================================

    from app.services.matching_engine import (
        match_candidate
    )

    from app.services.shap_explainer import (
        explain_candidate
    )

    ranked_candidates = []

    # ========================================================
    # Process every candidate
    # ========================================================

    for candidate_data in request.candidates:

        candidate = candidate_data.candidate

        # ----------------------------------------------------
        # Matching engine
        # ----------------------------------------------------

        result = match_candidate(
            job=request.job,
            candidate=candidate.model_dump(),
            resume_text=candidate_data.resume_text
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        skill_score = result[
            "skill_score"
        ]

        preferred_skill_score = result[
            "preferred_skill_score"
        ]

        experience_score = result[
            "experience_score"
        ]

        semantic_score = result[
            "semantic_score"
        ]

        overall_score = result[
            "overall_score"
        ]

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        try:

            shap_explanation = explain_candidate(
                skill_score=skill_score,
                preferred_skill_score=preferred_skill_score,
                experience_score=experience_score,
                semantic_score=semantic_score
            )

        except Exception as shap_error:

            shap_explanation = {
                "base_score": None,
                "final_score": overall_score,
                "features": [],
                "top_positive_factors": [],
                "top_negative_factors": [],
                "error": (
                    "SHAP explanation failed: "
                    f"{str(shap_error)}"
                )
            }

        # ----------------------------------------------------
        # Candidate result
        # ----------------------------------------------------

        ranked_candidates.append({

            "candidate_name": (
                candidate.candidate_name
            ),

            "email": (
                candidate.email
            ),

            "phone": (
                candidate.phone
            ),

            "overall_score": (
                overall_score
            ),

            "skill_score": (
                skill_score
            ),

            "preferred_skill_score": (
                preferred_skill_score
            ),

            "experience_score": (
                experience_score
            ),

            "semantic_score": (
                semantic_score
            ),

            "matched_required_skills": (
                result[
                    "matched_required_skills"
                ]
            ),

            "missing_required_skills": (
                result[
                    "missing_required_skills"
                ]
            ),

            "matched_preferred_skills": (
                result[
                    "matched_preferred_skills"
                ]
            ),

            "missing_preferred_skills": (
                result[
                    "missing_preferred_skills"
                ]
            ),

            "recommendation": (
                result[
                    "recommendation"
                ]
            ),

            "shap_explanation": (
                shap_explanation
            )
        })

    # ========================================================
    # Sort
    # ========================================================

    ranked_candidates.sort(
        key=lambda candidate: candidate[
            "overall_score"
        ],
        reverse=True
    )

    # ========================================================
    # Assign rank
    # ========================================================

    for index, candidate in enumerate(
        ranked_candidates,
        start=1
    ):

        candidate["rank"] = index

    # ========================================================
    # Response
    # ========================================================

    return {

        "message": (
            "Candidates ranked successfully"
        ),

        "total_candidates": (
            len(ranked_candidates)
        ),

        "rankings": ranked_candidates
    }