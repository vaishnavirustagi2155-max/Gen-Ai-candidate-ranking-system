from typing import Annotated
import inspect
import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.api.resumes import (
    extract_resume_text,
    analyze_resume,
)

from app.api.jobs import (
    analyze_job,
    JobDescriptionRequest,
)

from app.api.ranking import (
    rank_candidates,
    RankingRequest,
    CandidateForRanking,
)


router = APIRouter(
    prefix="/api/ranking",
    tags=["End-to-End Ranking"],
)


# ============================================================
# END-TO-END RANKING
# ============================================================

@router.post("/process")
async def process_ranking(
    job_description: Annotated[
        str,
        Form(...)
    ],

    files: Annotated[
        list[UploadFile],
        File(...)
    ],
):
    """
    Complete end-to-end candidate ranking.

    Accepts multiple PDF/DOCX resumes.
    There is no hard-coded 10 or 20 resume limit.
    """

    # ========================================================
    # 1. Validate Job Description
    # ========================================================

    if not job_description.strip():

        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    # ========================================================
    # 2. Validate Files
    # ========================================================

    if not files:

        raise HTTPException(
            status_code=400,
            detail="At least one resume is required.",
        )

    # ========================================================
    # 3. Analyze Job Description
    # ========================================================

    try:

        job_request = JobDescriptionRequest(
            job_description=job_description
        )

        job_result = analyze_job(
            job_request
        )

        if inspect.isawaitable(job_result):

            job_result = await job_result

        # The structured job is returned inside "job"
        job = job_result.get(
            "job",
            job_result
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Job description analysis failed: "
                f"{str(e)}"
            ),
        )

    # ========================================================
    # 4. Process Resumes
    # ========================================================

    candidate_objects = []

    successful_files = []

    failed_files = []

    temporary_files = []

    try:

        for uploaded_file in files:

            filename = (
                uploaded_file.filename
                or "resume"
            )

            extension = os.path.splitext(
                filename
            )[1].lower()

            # ------------------------------------------------
            # Validate file type
            # ------------------------------------------------

            if extension not in [
                ".pdf",
                ".docx",
            ]:

                failed_files.append({
                    "filename": filename,
                    "error": (
                        "Unsupported file type. "
                        "Only PDF and DOCX files "
                        "are allowed."
                    ),
                })

                continue

            try:

                # --------------------------------------------
                # Read file
                # --------------------------------------------

                file_content = (
                    await uploaded_file.read()
                )

                if not file_content:

                    failed_files.append({
                        "filename": filename,
                        "error": "Empty file.",
                    })

                    continue

                # --------------------------------------------
                # Save temporary file
                # --------------------------------------------

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension,
                )

                temp_path = temp_file.name

                try:

                    temp_file.write(
                        file_content
                    )

                finally:

                    temp_file.close()

                temporary_files.append(
                    temp_path
                )

                # --------------------------------------------
                # Extract resume text
                # --------------------------------------------

                resume_text = extract_resume_text(
                    temp_path
                )

                if not resume_text:

                    failed_files.append({
                        "filename": filename,
                        "error": (
                            "Could not extract "
                            "resume text."
                        ),
                    })

                    continue

                if len(
                    resume_text.strip()
                ) < 20:

                    failed_files.append({
                        "filename": filename,
                        "error": (
                            "Resume text is too "
                            "short to analyze."
                        ),
                    })

                    continue

                # --------------------------------------------
                # Analyze candidate
                # --------------------------------------------

                candidate = analyze_resume(
                    resume_text
                )

                if inspect.isawaitable(
                    candidate
                ):

                    candidate = await candidate

                # --------------------------------------------
                # Create ranking object
                # --------------------------------------------

                candidate_objects.append(
                    CandidateForRanking(
                        candidate=candidate,
                        resume_text=resume_text,
                    )
                )

                successful_files.append(
                    filename
                )

            except Exception as e:

                failed_files.append({
                    "filename": filename,
                    "error": str(e),
                })

    finally:

        # ====================================================
        # Delete temporary files
        # ====================================================

        for temp_path in temporary_files:

            try:

                if os.path.exists(temp_path):

                    os.remove(temp_path)

            except Exception:

                pass

    # ========================================================
    # 5. Validate Candidates
    # ========================================================

    if not candidate_objects:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "No valid candidates "
                    "could be processed."
                ),
                "failed_files": failed_files,
            },
        )

    # ========================================================
    # 6. Rank Candidates
    # ========================================================

    try:

        ranking_request = RankingRequest(
            job=job,
            candidates=candidate_objects,
        )

        ranking_result = rank_candidates(
            ranking_request
        )

        if inspect.isawaitable(
            ranking_result
        ):

            ranking_result = await ranking_result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Candidate ranking failed: "
                f"{str(e)}"
            ),
        )

    # ========================================================
    # 7. Final Response
    # ========================================================

    return {
        "message": (
            "Candidate ranking completed "
            "successfully"
        ),

        "job": job,

        "total_candidates": len(
            candidate_objects
        ),

        "successful_files": successful_files,

        "failed_files": failed_files,

        "rankings": ranking_result.get(
            "rankings",
            ranking_result,
        ),
    }