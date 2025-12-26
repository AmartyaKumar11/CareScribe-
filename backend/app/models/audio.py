from pydantic import BaseModel


class AudioUploadResponse(BaseModel):
    status: str

