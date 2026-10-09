
from typing import Annotated
import inspect
import logging
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ranking",
    tags=["End-to-End Ranking"],
)


@router.post("/process")
async def process_ranking(
    job_description: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
):
    if not job_description or not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one resume is required.",
        )

    logger.info(
        "Ranking request received with %s uploaded file(s).",
        len(files),
    )

    try:
        job_request = JobDescriptionRequest(
            job_description=job_description.strip()
        )

        job_result = analyze_job(job_request)

        if inspect.isawaitable(job_result):
            job_result = await job_result

        if not isinstance(job_result, dict):
            raise TypeError(
                "analyze_job() must return a dictionary."
            )

        job = job_result.get("job", job_result)

        if not job:
            raise ValueError(
                "Job description analysis returned an empty result."
            )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Job description analysis failed.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Job description analysis failed.",
                "error": str(exc),
            },
        ) from exc

    candidate_objects = []
    successful_files = []
    failed_files = []
    temporary_files = []

    try:
        for uploaded_file in files:
            filename = uploaded_file.filename or "resume"
            temp_path = None

            try:
                extension = os.path.splitext(filename)[1].lower()

                if extension not in {".pdf", ".docx"}:
                    failed_files.append({
                        "filename": filename,
                        "error": "Only PDF and DOCX files are supported.",
                    })
                    continue

                content = await uploaded_file.read()

                if not content:
                    failed_files.append({
                        "filename": filename,
                        "error": "The uploaded file is empty.",
                    })
                    continue

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    suffix=extension,
                ) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                temporary_files.append(temp_path)

                resume_text = extract_resume_text(temp_path)

                if inspect.isawaitable(resume_text):
                    resume_text = await resume_text

                if not isinstance(resume_text, str):
                    raise TypeError(
                        "Resume extraction did not return text."
                    )

                resume_text = resume_text.strip()

                if len(resume_text) < 20:
                    failed_files.append({
                        "filename": filename,
                        "error": (
                            "Could not extract enough text from "
                            "the resume. Check whether the PDF is "
                            "scanned or the document is empty."
                        ),
                    })
                    continue

                candidate = analyze_resume(resume_text)

                if inspect.isawaitable(candidate):
                    candidate = await candidate

                if candidate is None:
                    raise ValueError(
                        "Resume analysis returned no candidate."
                    )

                candidate_objects.append(
                    CandidateForRanking(
                        candidate=candidate,
                        resume_text=resume_text,
                    )
                )

                successful_files.append(filename)

                logger.info(
                    "Successfully processed resume: %s",
                    filename,
                )

            except Exception as exc:
                logger.exception(
                    "Resume processing failed for %s.",
                    filename,
                )

                failed_files.append({
                    "filename": filename,
                    "error": str(exc),
                })

            finally:
                await uploaded_file.close()

    finally:
        for temp_path in temporary_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                logger.exception(
                    "Failed to delete temporary file: %s",
                    temp_path,
                )

    if not candidate_objects:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid candidates could be processed.",
                "failed_files": failed_files,
            },
        )

    try:
        ranking_request = RankingRequest(
            job=job,
            candidates=candidate_objects,
        )

        ranking_result = rank_candidates(ranking_request)

        if inspect.isawaitable(ranking_result):
            ranking_result = await ranking_result

        if not isinstance(ranking_result, dict):
            raise TypeError(
                "rank_candidates() must return a dictionary."
            )

        rankings = ranking_result.get(
            "rankings",
            ranking_result,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Candidate ranking failed.")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Candidate ranking failed.",
                "error": str(exc),
            },
        ) from exc

    logger.info(
        "Ranking completed. Successful resumes: %s; failed resumes: %s.",
        len(successful_files),
        len(failed_files),
    )

    return {
        "message": "Candidate ranking completed successfully",
        "job": job,
        "total_candidates": len(candidate_objects),
        "successful_files": successful_files,
        "failed_files": failed_files,
        "rankings": rankings,
    }
