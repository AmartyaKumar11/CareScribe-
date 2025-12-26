from pydantic import BaseModel


class SessionCreateResponse(BaseModel):
    session_id: str
    status: str

