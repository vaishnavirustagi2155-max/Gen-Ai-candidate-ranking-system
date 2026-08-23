from fastapi import APIRouter


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)


@router.get("/test")
def auth_test():
    return {
        "message": "Authentication router is working"
    }