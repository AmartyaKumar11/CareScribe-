from fastapi import APIRouter, HTTPException
from uuid import uuid4
from app.models.session import SessionCreateResponse

router = APIRouter()


@router.post("", response_model=SessionCreateResponse, status_code=201)
async def create_session():
    """
    Create a new consultation session.
    Returns a UUID for the session.
    """
    session_id = str(uuid4())
    
    return SessionCreateResponse(
        session_id=session_id,
        status="created"
    )

