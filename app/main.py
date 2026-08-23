from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    auth,
    jobs,
    resumes,
    ranking,
    reports,
    process,
)


app = FastAPI(
    title="GenAI Candidate Ranking System",
    description="AI-powered candidate ranking and recruitment assistant",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://kram-omega.vercel.app",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# Routers
# ============================================================

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(resumes.router)
app.include_router(ranking.router)
app.include_router(reports.router)
app.include_router(process.router)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "message": "GenAI Candidate Ranking System API is running",
        "version": "1.0.0",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
    }


# ============================================================
# Custom OpenAPI
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # --------------------------------------------------------
    # Fix generated schema for multiple-resume upload endpoint
    # --------------------------------------------------------

    process_schema_name = (
        "Body_process_ranking_api_ranking_process_post"
    )

    schemas = (
        openapi_schema
        .get("components", {})
        .get("schemas", {})
    )

    process_schema = schemas.get(
        process_schema_name
    )

    if process_schema:

        properties = process_schema.get(
            "properties",
            {}
        )

        files_property = properties.get(
            "files"
        )

        if files_property:

            files_property.clear()

            files_property.update({
                "type": "array",
                "title": "Files",
                "description": (
                    "Upload one or more PDF or DOCX "
                    "resume files."
                ),
                "items": {
                    "type": "string",
                    "format": "binary",
                },
            })

    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi