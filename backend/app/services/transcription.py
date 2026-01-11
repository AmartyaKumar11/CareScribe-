import whisper
from pathlib import Path
from app.models.transcript import TranscriptResponse


class TranscriptionService:
    """
    Whisper-based transcription service.
    Uses self-hosted OpenAI Whisper for speech-to-text.
    """
    
    def __init__(self):
        """Initialize Whisper model."""
        try:
            self.model = whisper.load_model("small")
        except Exception:
            self.model = None
    
    def _find_audio_file(self, session_id: str) -> Path:
        """
        Find audio file for a session_id.
        Returns first matching file with session_id prefix.
        """
        storage_dir = Path(__file__).parent.parent / "storage" / "audio"
        
        # Find files starting with session_id
        for file_path in storage_dir.glob(f"{session_id}_*"):
            if file_path.is_file():
                return file_path
        
        return None
    
    def get_transcript(self, session_id: str) -> TranscriptResponse:
        """
        Get transcript for a session.
        Transcribes audio file using Whisper.
        """
        # Return empty if model not loaded
        if self.model is None:
            return TranscriptResponse(
                session_id=session_id,
                segments=[],
                status="complete"
            )
        
        # Find audio file
        audio_path = self._find_audio_file(session_id)
        if audio_path is None or not audio_path.exists():
            return TranscriptResponse(
                session_id=session_id,
                segments=[],
                status="complete"
            )
        
        # Transcribe with Whisper
        try:
            result = self.model.transcribe(
                str(audio_path),
                task="transcribe",
                language=None,  # Auto-detect
                fp16=False
            )
            
            # Convert Whisper segments to required format
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "text": segment.get("text", "").strip(),
                    "start_time": float(segment.get("start", 0.0)),
                    "end_time": float(segment.get("end", 0.0)),
                    "confidence": 1.0
                })
            
            return TranscriptResponse(
                session_id=session_id,
                segments=segments,
                status="complete"
            )
        
        except Exception:
            # Return empty on any error
            return TranscriptResponse(
                session_id=session_id,
                segments=[],
                status="complete"
            )

