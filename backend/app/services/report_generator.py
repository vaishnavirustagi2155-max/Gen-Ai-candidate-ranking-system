from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def generate_ranking_report(
    job: dict,
    rankings: list[dict]
) -> BytesIO:

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
    )

    story = []

    # ------------------------------------------------
    # Title
    # ------------------------------------------------

    story.append(
        Paragraph(
            "GenAI Candidate Ranking Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Generated: "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    # ------------------------------------------------
    # Job information
    # ------------------------------------------------

    story.append(
        Paragraph(
            "Job Information",
            heading_style
        )
    )

    job_title = job.get(
        "job_title",
        "Untitled Position"
    )

    job_description = job.get(
        "job_description",
        ""
    )

    job_info = [
        ["Job Title", job_title],
        [
            "Candidates Evaluated",
            str(len(rankings))
        ],
        [
            "Required Skills",
            ", ".join(
                job.get("required_skills", [])
            )
        ],
        [
            "Preferred Skills",
            ", ".join(
                job.get("preferred_skills", [])
            )
        ],
    ]

    job_table = Table(
        job_info,
        colWidths=[45 * mm, 135 * mm]
    )

    job_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(job_table)

    story.append(Spacer(1, 12))

    # ------------------------------------------------
    # Job description
    # ------------------------------------------------

    story.append(
        Paragraph(
            "Job Description",
            heading_style
        )
    )

    story.append(
        Paragraph(
            job_description,
            normal_style
        )
    )

    # ------------------------------------------------
    # Ranking table
    # ------------------------------------------------

    story.append(
        Paragraph(
            "Candidate Rankings",
            heading_style
        )
    )

    ranking_data = [
        [
            "Rank",
            "Candidate",
            "Score",
            "Recommendation"
        ]
    ]

    for candidate in rankings:

        ranking_data.append([
            str(candidate.get("rank", "")),
            candidate.get(
                "candidate_name",
                "Unknown"
            ),
            f'{candidate.get("overall_score", 0):.2f}%',
            candidate.get(
                "recommendation",
                ""
            )
        ])

    ranking_table = Table(
        ranking_data,
        colWidths=[
            15 * mm,
            65 * mm,
            30 * mm,
            60 * mm
        ],
        repeatRows=1
    )

    ranking_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.darkgrey
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ALIGN",
                (0, 0),
                (0, -1),
                "CENTER"
            ),
            (
                "ALIGN",
                (2, 1),
                (2, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.whitesmoke
                ]
            ),
        ])
    )

    story.append(ranking_table)

    # ------------------------------------------------
    # Detailed candidate analysis
    # ------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Detailed Candidate Analysis",
            heading_style
        )
    )

    for candidate in rankings:

        candidate_name = candidate.get(
            "candidate_name",
            "Unknown Candidate"
        )

        story.append(
            Paragraph(
                f"Rank #{candidate.get('rank', '')} — "
                f"{candidate_name}",
                heading_style
            )
        )

        details = [
            [
                "Overall Score",
                f'{candidate.get("overall_score", 0):.2f}%'
            ],
            [
                "Required Skill Score",
                f'{candidate.get("skill_score", 0):.2f}%'
            ],
            [
                "Preferred Skill Score",
                f'{candidate.get("preferred_skill_score", 0):.2f}%'
            ],
            [
                "Experience Score",
                f'{candidate.get("experience_score", 0):.2f}%'
            ],
            [
                "Semantic Fit",
                f'{candidate.get("semantic_score", 0):.2f}%'
            ],
            [
                "Recommendation",
                candidate.get(
                    "recommendation",
                    ""
                )
            ],
        ]

        details_table = Table(
            details,
            colWidths=[
                55 * mm,
                115 * mm
            ]
        )

        details_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
            ])
        )

        story.append(details_table)
        story.append(Spacer(1, 6))

        matched_required = candidate.get(
            "matched_required_skills",
            []
        )

        missing_required = candidate.get(
            "missing_required_skills",
            []
        )

        matched_preferred = candidate.get(
            "matched_preferred_skills",
            []
        )

        missing_preferred = candidate.get(
            "missing_preferred_skills",
            []
        )

        story.append(
            Paragraph(
                "<b>Matched Required Skills:</b> "
                + (
                    ", ".join(matched_required)
                    if matched_required
                    else "None"
                ),
                small_style
            )
        )

        story.append(
            Paragraph(
                "<b>Missing Required Skills:</b> "
                + (
                    ", ".join(missing_required)
                    if missing_required
                    else "None"
                ),
                small_style
            )
        )

        story.append(
            Paragraph(
                "<b>Matched Preferred Skills:</b> "
                + (
                    ", ".join(matched_preferred)
                    if matched_preferred
                    else "None"
                ),
                small_style
            )
        )

        story.append(
            Paragraph(
                "<b>Missing Preferred Skills:</b> "
                + (
                    ", ".join(missing_preferred)
                    if missing_preferred
                    else "None"
                ),
                small_style
            )
        )

        story.append(Spacer(1, 12))

    # ------------------------------------------------
    # Build PDF
    # ------------------------------------------------

    document.build(story)

    buffer.seek(0)

    return buffer