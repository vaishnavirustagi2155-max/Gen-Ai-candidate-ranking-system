from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.resume_analyzer import analyze_resume
from app.services.resume_parser import extract_resume_text


router = APIRouter(
    prefix="/api/resumes",
    tags=["Resumes"]
)


TEMP_DIR = Path("temp/resumes")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(
        ...,
        description="Upload a PDF or DOCX resume"
    )
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    file_id = str(uuid4())

    saved_filename = f"{file_id}{extension}"

    file_path = TEMP_DIR / saved_filename

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    try:
        extracted_text = extract_resume_text(
            str(file_path)
        )

    except Exception as error:

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Could not process resume: {str(error)}"
        )

    if not extracted_text:

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the resume."
        )

    candidate = analyze_resume(
        extracted_text
    )

    return {
        "message": "Resume analyzed successfully",
        "resume_id": file_id,
        "original_filename": file.filename,
        "candidate": candidate
    }