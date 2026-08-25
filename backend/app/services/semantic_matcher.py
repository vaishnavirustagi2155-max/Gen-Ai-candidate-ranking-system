"""
Lightweight semantic similarity service.

Render-safe implementation.

This version intentionally does NOT use:
- SentenceTransformer
- PyTorch
- Hugging Face Hub

Instead it uses:
- TF-IDF
- cosine similarity

This keeps memory usage very low while preserving
semantic-style text similarity for candidate ranking.
"""

from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(text: str | None) -> str:
    """
    Safely normalize text.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text:
        return ""

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# SKILL CLEANING
# ============================================================

def _clean_skills(
    skills: Iterable[str] | None
) -> list[str]:
    """
    Clean candidate skills.
    """

    if not skills:
        return []

    cleaned = []

    for skill in skills:

        if skill is None:
            continue

        skill = _clean_text(skill)

        if skill:
            cleaned.append(skill)

    return cleaned


# ============================================================
# TEXT SIMILARITY
# ============================================================

def _calculate_text_similarity(
    text_a: str,
    text_b: str
) -> float:
    """
    Calculate cosine similarity using TF-IDF.

    Returns:
        Score from 0 to 100.
    """

    text_a = _clean_text(text_a)
    text_b = _clean_text(text_b)

    if not text_a or not text_b:
        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            [text_a, text_b]
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        score = float(similarity) * 100.0

        return round(
            max(
                0.0,
                min(score, 100.0)
            ),
            2
        )

    except Exception as error:

        print(
            "Semantic similarity calculation failed: "
            f"{error}"
        )

        return 0.0


# ============================================================
# PUBLIC SIMILARITY FUNCTION
# ============================================================

def similarity_score(
    text_a: str,
    text_b: str
) -> float:
    """
    Public similarity function.

    Returns:
        Score from 0 to 100.
    """

    return _calculate_text_similarity(
        text_a,
        text_b
    )


# ============================================================
# EMBEDDING COMPATIBILITY FUNCTION
# ============================================================

def create_embedding(
    text: str
):
    """
    Compatibility function.

    The old implementation returned SentenceTransformer
    embeddings.

    This Render-safe implementation intentionally does
    not expose heavyweight embeddings.

    It returns None because the application should use
    calculate_semantic_similarity() instead.
    """

    return None


# ============================================================
# MULTI-TEXT COMPATIBILITY FUNCTION
# ============================================================

def _create_embeddings(
    texts: list[str]
):
    """
    Compatibility function.

    Kept so older imports do not immediately break.

    Heavy embedding generation has intentionally been removed.
    """

    return None


# ============================================================
# MAIN SEMANTIC MATCHING
# ============================================================

def calculate_semantic_similarity(
    job_description: str,
    resume_text: str,
    candidate_skills: list[str] | None = None
) -> float:
    """
    Calculate lightweight semantic similarity between
    a job description and a resume.

    Formula:

        60% JD vs Resume
        +
        40% JD vs Candidate Skills

    Returns:
        Semantic score from 0 to 100.
    """

    job_description = _clean_text(
        job_description
    )

    resume_text = _clean_text(
        resume_text
    )

    if not job_description:
        return 0.0

    if not resume_text:
        return 0.0

    # --------------------------------------------------------
    # Candidate skills
    # --------------------------------------------------------

    cleaned_skills = _clean_skills(
        candidate_skills
    )

    skills_text = " ".join(
        cleaned_skills
    )

    # --------------------------------------------------------
    # Full JD vs Resume
    # --------------------------------------------------------

    full_resume_score = (
        _calculate_text_similarity(
            job_description,
            resume_text
        )
    )

    # --------------------------------------------------------
    # JD vs Candidate Skills
    # --------------------------------------------------------

    if skills_text:

        skill_semantic_score = (
            _calculate_text_similarity(
                job_description,
                skills_text
            )
        )

    else:

        skill_semantic_score = 0.0

    # --------------------------------------------------------
    # Weighted score
    # --------------------------------------------------------

    semantic_score = (
        full_resume_score * 0.60
        +
        skill_semantic_score * 0.40
    )

    return round(
        max(
            0.0,
            min(
                semantic_score,
                100.0
            )
        ),
        2
    )


# ============================================================
# MODEL COMPATIBILITY
# ============================================================

def preload_model() -> None:
    """
    Compatibility function.

    There is no model to preload in the lightweight
    implementation.
    """

    return None


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status() -> dict:
    """
    Return semantic engine status.
    """

    return {
        "loaded": True,
        "model": "TF-IDF",
        "backend": "scikit-learn",
        "remote_model_download": False,
        "huggingface_required": False,
        "memory_safe": True,
    }