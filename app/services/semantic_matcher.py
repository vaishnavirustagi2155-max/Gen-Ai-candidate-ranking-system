from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embedding(text: str):
    return model.encode(
        text,
        convert_to_numpy=True
    )


def similarity_score(
    text_a: str,
    text_b: str
) -> float:

    if not text_a.strip() or not text_b.strip():
        return 0.0

    embedding_a = create_embedding(text_a)
    embedding_b = create_embedding(text_b)

    similarity = cosine_similarity(
        [embedding_a],
        [embedding_b]
    )[0][0]

    return round(
        max(0.0, min(float(similarity) * 100, 100.0)),
        2
    )


def calculate_semantic_similarity(
    job_description: str,
    resume_text: str,
    candidate_skills: list[str] | None = None
) -> float:

    if not job_description.strip():
        return 0.0

    if not resume_text.strip():
        return 0.0

    candidate_skills = candidate_skills or []

    # ----------------------------------------
    # 1. Full JD vs resume
    # ----------------------------------------

    full_resume_score = similarity_score(
        job_description,
        resume_text
    )

    # ----------------------------------------
    # 2. JD vs candidate skills
    # ----------------------------------------

    skills_text = ", ".join(candidate_skills)

    skill_semantic_score = similarity_score(
        job_description,
        skills_text
    ) if skills_text else 0.0

    # ----------------------------------------
    # 3. Weighted semantic score
    # ----------------------------------------

    semantic_score = (
        full_resume_score * 0.60
        + skill_semantic_score * 0.40
    )

    return round(
        semantic_score,
        2
    )