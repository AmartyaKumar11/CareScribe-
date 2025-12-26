from app.models.transcript import TranscriptResponse


class TranscriptionService:
    """
    Stub transcription service.
    Returns hardcoded transcript segments for development.
    """
    
    def get_transcript(self, session_id: str) -> TranscriptResponse:
        """
        Get transcript for a session.
        Returns stubbed data in the required format.
        """
        # Stubbed transcript segments
        segments = [
            {
                "text": "parso se halka fever shuru hua tha",
                "start_time": 0.0,
                "end_time": 4.2,
                "confidence": 0.85
            },
            {
                "text": "aur abhi tak theek nahi hua",
                "start_time": 4.2,
                "end_time": 7.8,
                "confidence": 0.82
            }
        ]
        
        return TranscriptResponse(
            session_id=session_id,
            segments=segments,
            status="complete"
        )

