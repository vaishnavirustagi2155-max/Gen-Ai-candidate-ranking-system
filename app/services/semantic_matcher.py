"""
KRAM AI - Low-Memory Semantic Matching Service

Designed for:
- Render low-memory deployment
- FastAPI
- CPU-only inference
- Lazy SentenceTransformer loading
- Graceful Hugging Face/model failures
- Single model instance per process

IMPORTANT:
The semantic model is NOT loaded when this module is imported.
It is loaded only when semantic matching is actually requested.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import numpy as np


# ============================================================
# LOW-MEMORY ENVIRONMENT SETTINGS
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prevent Hugging Face from using unnecessary parallelism.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL",
    "all-MiniLM-L6-v2"
)

# Very small batch for Render.
BATCH_SIZE = 1

# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
EMBEDDING_DIMENSION = 384


# ============================================================
# MODEL STATE
# ============================================================

_model_failed = False
_model_error: Optional[str] = None


# ============================================================
# SAFE ZERO EMBEDDING
# ============================================================

def _zero_embedding() -> np.ndarray:
    """
    Return a zero vector when semantic model is unavailable.
    """

    return np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float32
    )


# ============================================================
# LAZY MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def get_model():
    """
    Load SentenceTransformer lazily.

    The model is loaded only when semantic matching
    is actually requested.

    If model loading fails, None is returned instead
    of crashing the FastAPI application.
    """

    global _model_failed
    global _model_error

    # --------------------------------------------------------
    # Don't repeatedly attempt a failed model download.
    # --------------------------------------------------------

    if _model_failed:

        logger.warning(
            "Semantic model previously failed to load. "
            "Semantic matching is disabled for this process."
        )

        return None

    try:

        # ----------------------------------------------------
        # Import PyTorch only when required.
        # ----------------------------------------------------

        import torch

        # ----------------------------------------------------
        # Limit CPU threads.
        # ----------------------------------------------------

        try:

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        except RuntimeError:

            # Thread configuration may already be initialized.
            pass

        # ----------------------------------------------------
        # Import SentenceTransformer lazily.
        # ----------------------------------------------------

        from sentence_transformers import (
            SentenceTransformer
        )

        logger.info(
            "Loading semantic model: %s",
            MODEL_NAME
        )

        # ----------------------------------------------------
        # Load CPU model.
        # ----------------------------------------------------

        model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        # ----------------------------------------------------
        # Evaluation mode.
        # ----------------------------------------------------

        try:

            model.eval()

        except AttributeError:

            pass

        logger.info(
            "Semantic model loaded successfully."
        )

        return model

    except Exception as exc:

        _model_failed = True

        _model_error = str(exc)

        logger.exception(
            "Semantic model failed to load. "
            "Semantic matching will be disabled. "
            "Error: %s",
            exc
        )

        return None


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(
    text: Optional[str]
) -> str:
    """
    Safely normalize text.
    """

    if text is None:

        return ""

    if not isinstance(text, str):

        text = str(text)

    return " ".join(
        text.strip().split()
    )


# ============================================================
# SINGLE EMBEDDING
# ============================================================

def create_embedding(
    text: str
) -> np.ndarray:
    """
    Create a single normalized embedding.

    If the semantic model is unavailable,
    a zero vector is returned.
    """

    text = _clean_text(text)

    if not text:

        return _zero_embedding()

    model = get_model()

    # --------------------------------------------------------
    # Model unavailable.
    # --------------------------------------------------------

    if model is None:

        return _zero_embedding()

    try:

        embedding = model.encode(
            text,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(
            embedding,
            dtype=np.float32
        )

    except Exception as exc:

        logger.exception(
            "Semantic embedding generation failed: %s",
            exc
        )

        return _zero_embedding()


# ============================================================
# MULTIPLE EMBEDDINGS
# ============================================================

def _create_embeddings(
    texts: list[str]
) -> np.ndarray:
    """
    Generate embeddings for multiple texts.

    Uses a very small batch to reduce memory usage.
    """

    cleaned_texts = [
        _clean_text(text)
        for text in texts
    ]

    # --------------------------------------------------------
    # Empty input.
    # --------------------------------------------------------

    if not cleaned_texts:

        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # If everything is empty.
    # --------------------------------------------------------

    if not any(cleaned_texts):

        return np.zeros(
            (
                len(cleaned_texts),
                EMBEDDING_DIMENSION
            ),
            dtype=np.float32
        )

    model = get_model()

    # --------------------------------------------------------
    # Model unavailable.
    # --------------------------------------------------------

    if model is None:

        return np.zeros(
            (
                len(cleaned_texts),
                EMBEDDING_DIMENSION
            ),
            dtype=np.float32
        )

    try:

        embeddings = model.encode(
            cleaned_texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    except Exception as exc:

        logger.exception(
            "Batch semantic embedding failed: %s",
            exc
        )

        return np.zeros(
            (
                len(cleaned_texts),
                EMBEDDING_DIMENSION
            ),
            dtype=np.float32
        )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Lightweight cosine similarity.

    Avoids sklearn overhead for this simple calculation.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    )

    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )

    if norm_a == 0.0 or norm_b == 0.0:

        return 0.0

    similarity = float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        (norm_a * norm_b)
    )

    return max(
        -1.0,
        min(
            similarity,
            1.0
        )
    )


# ============================================================
# SIMPLE SIMILARITY SCORE
# ============================================================

def similarity_score(
    text_a: str,
    text_b: str
) -> float:
    """
    Calculate semantic similarity between two texts.

    Returns:
        0 to 100
    """

    text_a = _clean_text(
        text_a
    )

    text_b = _clean_text(
        text_b
    )

    if not text_a or not text_b:

        return 0.0

    # --------------------------------------------------------
    # Check whether model is available.
    # --------------------------------------------------------

    model = get_model()

    if model is None:

        logger.warning(
            "Semantic similarity unavailable. "
            "Returning 0.0."
        )

        return 0.0

    embedding_a = create_embedding(
        text_a
    )

    embedding_b = create_embedding(
        text_b
    )

    similarity = _cosine_similarity(
        embedding_a,
        embedding_b
    )

    # --------------------------------------------------------
    # Convert [-1, 1] to percentage.
    # --------------------------------------------------------

    score = similarity * 100.0

    # --------------------------------------------------------
    # Keep score in safe range.
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            score,
            100.0
        )
    )

    return round(
        score,
        2
    )


# ============================================================
# MAIN SEMANTIC MATCHING FUNCTION
# ============================================================

def calculate_semantic_similarity(
    job_description: str,
    resume_text: str,
    candidate_skills: Optional[list[str]] = None
) -> float:
    """
    Calculate semantic similarity between a job description
    and a candidate resume.

    Formula:

        60% Full JD vs Resume
        +
        40% JD vs Candidate Skills

    Returns:
        Semantic score from 0 to 100.

    IMPORTANT:
    If the Hugging Face model cannot be loaded,
    this function returns 0 instead of crashing
    the entire ranking API.
    """

    # ========================================================
    # CLEAN INPUT
    # ========================================================

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


    # ========================================================
    # MODEL CHECK
    # ========================================================

    model = get_model()

    if model is None:

        logger.warning(
            "Semantic model unavailable. "
            "Continuing ranking without semantic score."
        )

        return 0.0


    # ========================================================
    # CANDIDATE SKILLS
    # ========================================================

    if candidate_skills is None:

        candidate_skills = []

    cleaned_skills = []

    for skill in candidate_skills:

        if skill is None:

            continue

        skill = _clean_text(
            skill
        )

        if skill:

            cleaned_skills.append(
                skill
            )


    # ========================================================
    # SKILLS TEXT
    # ========================================================

    skills_text = ", ".join(
        cleaned_skills
    )


    # ========================================================
    # PREPARE TEXTS
    # ========================================================

    texts = [
        job_description,
        resume_text
    ]

    if skills_text:

        texts.append(
            skills_text
        )


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    embeddings = _create_embeddings(
        texts
    )

    if embeddings.shape[0] < 2:

        return 0.0


    # ========================================================
    # JOB / RESUME
    # ========================================================

    job_embedding = embeddings[0]

    resume_embedding = embeddings[1]


    # ========================================================
    # FULL JD VS RESUME
    # ========================================================

    full_resume_similarity = (
        _cosine_similarity(
            job_embedding,
            resume_embedding
        )
    )

    full_resume_score = (
        max(
            0.0,
            min(
                full_resume_similarity * 100.0,
                100.0
            )
        )
    )


    # ========================================================
    # JD VS CANDIDATE SKILLS
    # ========================================================

    if skills_text and embeddings.shape[0] >= 3:

        skills_embedding = embeddings[2]

        skill_similarity = (
            _cosine_similarity(
                job_embedding,
                skills_embedding
            )
        )

        skill_semantic_score = (
            max(
                0.0,
                min(
                    skill_similarity * 100.0,
                    100.0
                )
            )
        )

    else:

        skill_semantic_score = 0.0


    # ========================================================
    # FINAL SEMANTIC SCORE
    # ========================================================

    semantic_score = (
        full_resume_score * 0.60
        +
        skill_semantic_score * 0.40
    )


    # ========================================================
    # FINAL SAFETY
    # ========================================================

    semantic_score = max(
        0.0,
        min(
            semantic_score,
            100.0
        )
    )


    return round(
        semantic_score,
        2
    )


# ============================================================
# MODEL PRELOAD
# ============================================================

def preload_model() -> None:
    """
    Explicitly load the semantic model.

    DO NOT call this during FastAPI startup on
    a low-memory Render instance.

    This function is provided for optional use
    on larger deployment instances.
    """

    get_model()


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status() -> dict:
    """
    Return the current semantic model status.

    Useful for debugging Render deployments.
    """

    if _model_failed:

        return {
            "available": False,
            "model": MODEL_NAME,
            "error": _model_error,
        }

    model = get_model()

    if model is None:

        return {
            "available": False,
            "model": MODEL_NAME,
            "error": _model_error,
        }

    return {
        "available": True,
        "model": MODEL_NAME,
        "error": None,
    }