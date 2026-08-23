"""
Low-memory semantic matching service.

Designed for Render's low-memory environment.

Key optimizations:
- SentenceTransformer is loaded lazily, not during module import.
- PyTorch CPU threads are limited.
- Model is loaded only when semantic matching is actually requested.
- Embeddings are generated in small batches.
- No gradients are used during inference.
- The model is cached after first use.
"""

import os

# ---------------------------------------------------------
# LOW-MEMORY / CPU SETTINGS
# ---------------------------------------------------------

# Limit CPU thread usage.
# This helps prevent excessive memory consumption on Render.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from functools import lru_cache

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL",
    "all-MiniLM-L6-v2"
)

# Small batch size keeps memory usage low.
BATCH_SIZE = 1


# ---------------------------------------------------------
# LAZY MODEL LOADING
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_model():
    """
    Load the SentenceTransformer model only when required.

    The model is cached so it is loaded only once per process.
    """

    # Import torch only when the model is actually needed.
    import torch

    # Limit PyTorch CPU threads.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Thread settings may already have been initialized.
        pass

    from sentence_transformers import SentenceTransformer

    print(f"Loading semantic model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    # Put the model in evaluation mode.
    try:
        model.eval()
    except AttributeError:
        pass

    print("Semantic model loaded successfully.")

    return model


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Safely normalize input text.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    return " ".join(text.strip().split())


# ---------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------

def create_embedding(text: str):
    """
    Create a single embedding using the cached model.

    Returns:
        numpy.ndarray
    """

    text = _clean_text(text)

    if not text:
        return np.zeros(384, dtype=np.float32)

    model = get_model()

    # SentenceTransformer performs inference without gradients.
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


# ---------------------------------------------------------
# SIMILARITY
# ---------------------------------------------------------

def similarity_score(
    text_a: str,
    text_b: str
) -> float:
    """
    Calculate semantic similarity between two texts.

    Returns:
        Score from 0 to 100.
    """

    text_a = _clean_text(text_a)
    text_b = _clean_text(text_b)

    if not text_a or not text_b:
        return 0.0

    embedding_a = create_embedding(text_a)
    embedding_b = create_embedding(text_b)

    similarity = cosine_similarity(
        embedding_a.reshape(1, -1),
        embedding_b.reshape(1, -1)
    )[0][0]

    # Convert [-1, 1] similarity to a safe percentage.
    score = float(similarity) * 100.0

    score = max(
        0.0,
        min(score, 100.0)
    )

    return round(score, 2)


# ---------------------------------------------------------
# OPTIMIZED MULTI-TEXT EMBEDDING
# ---------------------------------------------------------

def _create_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for multiple texts in one small batch.

    This is more memory-efficient than repeatedly loading
    model operations for each text.
    """

    cleaned_texts = [
        _clean_text(text)
        for text in texts
    ]

    if not any(cleaned_texts):
        return np.zeros(
            (len(cleaned_texts), 384),
            dtype=np.float32
        )

    model = get_model()

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


# ---------------------------------------------------------
# MAIN SEMANTIC MATCHING FUNCTION
# ---------------------------------------------------------

def calculate_semantic_similarity(
    job_description: str,
    resume_text: str,
    candidate_skills: list[str] | None = None
) -> float:
    """
    Calculate semantic similarity between a job description
    and a candidate resume.

    Formula:

        Final Semantic Score =
            60% Full JD vs Resume
            +
            40% JD vs Candidate Skills

    Returns:
        Semantic score from 0 to 100.
    """

    job_description = _clean_text(job_description)
    resume_text = _clean_text(resume_text)

    if not job_description:
        return 0.0

    if not resume_text:
        return 0.0

    # -----------------------------------------------------
    # Candidate skills
    # -----------------------------------------------------

    if candidate_skills is None:
        candidate_skills = []

    cleaned_skills = []

    for skill in candidate_skills:

        if skill is None:
            continue

        skill = _clean_text(skill)

        if skill:
            cleaned_skills.append(skill)

    skills_text = ", ".join(cleaned_skills)

    # -----------------------------------------------------
    # Generate embeddings together
    # -----------------------------------------------------

    texts = [
        job_description,
        resume_text
    ]

    if skills_text:
        texts.append(skills_text)

    embeddings = _create_embeddings(texts)

    job_embedding = embeddings[0]
    resume_embedding = embeddings[1]

    # -----------------------------------------------------
    # 1. Full JD vs Resume
    # -----------------------------------------------------

    full_resume_similarity = cosine_similarity(
        job_embedding.reshape(1, -1),
        resume_embedding.reshape(1, -1)
    )[0][0]

    full_resume_score = max(
        0.0,
        min(float(full_resume_similarity) * 100.0, 100.0)
    )

    # -----------------------------------------------------
    # 2. JD vs Candidate Skills
    # -----------------------------------------------------

    if skills_text:

        skills_embedding = embeddings[2]

        skill_similarity = cosine_similarity(
            job_embedding.reshape(1, -1),
            skills_embedding.reshape(1, -1)
        )[0][0]

        skill_semantic_score = max(
            0.0,
            min(float(skill_similarity) * 100.0, 100.0)
        )

    else:

        skill_semantic_score = 0.0

    # -----------------------------------------------------
    # 3. Weighted semantic score
    # -----------------------------------------------------

    semantic_score = (
        full_resume_score * 0.60
        +
        skill_semantic_score * 0.40
    )

    return round(
        max(
            0.0,
            min(semantic_score, 100.0)
        ),
        2
    )


# ---------------------------------------------------------
# OPTIONAL MODEL WARM-UP
# ---------------------------------------------------------

def preload_model() -> None:
    """
    Explicitly load the model.

    Normally you should NOT call this during FastAPI startup
    on a 512 MB Render instance.

    This function exists only if you later upgrade the
    Render instance or intentionally want model warm-up.
    """

    get_model()