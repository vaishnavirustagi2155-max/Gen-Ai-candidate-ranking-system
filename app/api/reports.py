from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.report_generator import generate_ranking_report


router = APIRouter(
    prefix="/api/reports",
    tags=["Reports"]
)


class RankingReportRequest(BaseModel):
    job: dict

    rankings: list[dict] = Field(
        ...,
        min_length=1
    )


@router.post("/download")
async def download_report(
    request: RankingReportRequest
):
    pdf_file = generate_ranking_report(
        job=request.job,
        rankings=request.rankings
    )

    return StreamingResponse(
        pdf_file,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                'attachment; filename="candidate_ranking_report.pdf"'
        }
    )