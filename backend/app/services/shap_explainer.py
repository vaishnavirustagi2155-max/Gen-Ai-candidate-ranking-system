from typing import Any

import numpy as np
import shap


# ============================================================
# Features used by the ranking engine
# ============================================================

FEATURE_NAMES = [
    "skill_score",
    "preferred_skill_score",
    "experience_score",
    "semantic_score",
]


# ============================================================
# Ranking weights
# Must match matching_engine.py
# ============================================================

WEIGHTS = np.array(
    [
        0.40,  # Required skills
        0.15,  # Preferred skills
        0.20,  # Experience
        0.25,  # Semantic similarity
    ],
    dtype=float,
)


# ============================================================
# Human-readable feature names
# ============================================================

FEATURE_LABELS = {
    "skill_score": "Required Skills",
    "preferred_skill_score": "Preferred Skills",
    "experience_score": "Experience",
    "semantic_score": "Semantic Similarity",
}


# ============================================================
# Ranking prediction function
# ============================================================

def ranking_prediction(
    X: np.ndarray,
) -> np.ndarray:

    X = np.asarray(
        X,
        dtype=float,
    )

    return (
        X[:, 0] * WEIGHTS[0]
        + X[:, 1] * WEIGHTS[1]
        + X[:, 2] * WEIGHTS[2]
        + X[:, 3] * WEIGHTS[3]
    )


# ============================================================
# SHAP background dataset
# ============================================================

BACKGROUND_DATA = np.array(
    [
        [0, 0, 0, 0],
        [25, 25, 25, 25],
        [50, 50, 50, 50],
        [75, 75, 75, 75],
        [100, 100, 100, 100],
    ],
    dtype=float,
)


# ============================================================
# SHAP Explainer
# ============================================================

explainer = shap.Explainer(
    ranking_prediction,
    BACKGROUND_DATA,
    feature_names=FEATURE_NAMES,
)


# ============================================================
# Explain a candidate
# ============================================================

def explain_candidate(
    skill_score: float,
    preferred_skill_score: float,
    experience_score: float,
    semantic_score: float,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Candidate feature values
    # --------------------------------------------------------

    values = np.array(
        [
            float(skill_score),
            float(preferred_skill_score),
            float(experience_score),
            float(semantic_score),
        ],
        dtype=float,
    )

    values_2d = values.reshape(1, -1)

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_result = explainer(values_2d)

    shap_values = np.asarray(
        shap_result.values
    )[0]

    # --------------------------------------------------------
    # Base value
    # --------------------------------------------------------

    base_values = np.asarray(
        shap_result.base_values
    ).reshape(-1)

    base_value = float(
        base_values[0]
    )

    # --------------------------------------------------------
    # Calculate final score
    # --------------------------------------------------------

    final_score = float(
        ranking_prediction(values_2d)[0]
    )

    # --------------------------------------------------------
    # Build feature explanations
    # --------------------------------------------------------

    contributions = []

    for index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        contribution = float(
            shap_values[index]
        )

        feature_value = float(
            values[index]
        )

        label = FEATURE_LABELS.get(
            feature_name,
            feature_name,
        )

        if contribution > 0:
            direction = "positive"

            reason = (
                f"{label} increased the "
                f"candidate's ranking score."
            )

        elif contribution < 0:
            direction = "negative"

            reason = (
                f"{label} reduced the "
                f"candidate's ranking score."
            )

        else:
            direction = "neutral"

            reason = (
                f"{label} had no impact on "
                f"the candidate's ranking score."
            )

        contributions.append(
            {
                "feature": label,
                "feature_key": feature_name,
                "value": round(
                    feature_value,
                    2,
                ),
                "shap_value": round(
                    contribution,
                    2,
                ),
                "direction": direction,
                "reason": reason,
            }
        )

    # --------------------------------------------------------
    # Sort by absolute SHAP impact
    # --------------------------------------------------------

    contributions.sort(
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Positive factors
    # --------------------------------------------------------

    positive_factors = [
        item
        for item in contributions
        if item["shap_value"] > 0
    ]

    # --------------------------------------------------------
    # Negative factors
    # --------------------------------------------------------

    negative_factors = [
        item
        for item in contributions
        if item["shap_value"] < 0
    ]

    # --------------------------------------------------------
    # Return explanation
    # --------------------------------------------------------

    return {
        "base_score": round(
            base_value,
            2,
        ),

        "final_score": round(
            final_score,
            2,
        ),

        "features": contributions,

        "top_positive_factors": (
            positive_factors[:3]
        ),

        "top_negative_factors": (
            negative_factors[:3]
        ),
    }