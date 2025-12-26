from pydantic import BaseModel
from typing import List


class TranscriptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    confidence: float


class TranscriptResponse(BaseModel):
    session_id: str
    segments: List[TranscriptSegment]
    status: str  # "complete" | "partial"

