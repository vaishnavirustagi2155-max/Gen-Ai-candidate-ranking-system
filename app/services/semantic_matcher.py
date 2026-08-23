from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# Model is NOT loaded when this module is imported.
# It will be loaded only when semantic matching is needed.
_model = None


# ============================================================
# Lazy model loader
# ============================================================

def get_model():
    """
    Load the SentenceTransformer model only when required.

    The model is cached after the first load so that it is
    reused for all subsequent candidates.
    """

    global _model

    if _model is None:

        print(
            f"Loading semantic model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        print(
            "Semantic model loaded successfully."
        )

    return _model


# ============================================================
# Create embedding
# ============================================================

def create_embedding(
    text: str
):
    """
    Convert text into an embedding vector.
    """

    if not text or not text.strip():
        return None

    model = get_model()

    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


# ============================================================
# Calculate similarity
# ============================================================

def similarity_score(
    text_a: str,
    text_b: str
) -> float:

    if not text_a or not text_a.strip():
        return 0.0

    if not text_b or not text_b.strip():
        return 0.0

    embedding_a = create_embedding(
        text_a
    )

    embedding_b = create_embedding(
        text_b
    )

    if embedding_a is None or embedding_b is None:
        return 0.0

    similarity = cosine_similarity(
        [embedding_a],
        [embedding_b]
    )[0][0]

    return round(
        max(
            0.0,
            min(
                float(similarity) * 100,
                100.0
            )
        ),
        2
    )


# ============================================================
# Calculate semantic similarity
# ============================================================

def calculate_semantic_similarity(
    job_description: str,
    resume_text: str,
    candidate_skills: list[str] | None = None
) -> float:

    if not job_description or not job_description.strip():
        return 0.0

    if not resume_text or not resume_text.strip():
        return 0.0

    candidate_skills = candidate_skills or []

    # ========================================================
    # 1. Full JD vs Resume
    # ========================================================

    full_resume_score = similarity_score(
        job_description,
        resume_text
    )

    # ========================================================
    # 2. JD vs Candidate Skills
    # ========================================================

    skills_text = ", ".join(
        candidate_skills
    )

    if skills_text:

        skill_semantic_score = similarity_score(
            job_description,
            skills_text
        )

    else:

        skill_semantic_score = 0.0

    # ========================================================
    # 3. Weighted Semantic Score
    #
    # Resume relevance = 60%
    # Skill relevance  = 40%
    # ========================================================

    semantic_score = (
        full_resume_score * 0.60
        + skill_semantic_score * 0.40
    )

    return round(
        semantic_score,
        2
    )