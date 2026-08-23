from pathlib import Path
from typing import Any
import uuid

import fitz
from docx import Document


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx"
}


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    """

    text_parts = []

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    try:
        for page in pdf:
            text_parts.append(
                page.get_text()
            )
    finally:
        pdf.close()

    return "\n".join(text_parts).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.
    """

    import io

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text.strip()
            )

    return "\n".join(paragraphs).strip()


def extract_resume_text(
    filename: str,
    file_bytes: bytes
) -> str:
    """
    Extract resume text based on file extension.
    """

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(
            file_bytes
        )

    if extension == ".docx":
        return extract_docx_text(
            file_bytes
        )

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Only PDF and DOCX files are supported."
    )


def generate_resume_id() -> str:
    return str(uuid.uuid4())