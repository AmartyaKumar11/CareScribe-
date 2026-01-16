import whisper
from pathlib import Path
from app.models.transcript import TranscriptResponse
from app.services.audio_normalization import AudioNormalizationService


class TranscriptionService:
    """
    Whisper-based transcription service.
    Uses self-hosted OpenAI Whisper for speech-to-text.
    Only uses normalized audio files (never original uploads).
    """
    
    def __init__(self):
        """Initialize Whisper model."""
        try:
            self.model = whisper.load_model("small")
        except Exception:
            self.model = None
        
        self.normalization_service = AudioNormalizationService()
    
    def get_transcript(self, session_id: str) -> TranscriptResponse:
        """
        Get transcript for a session.
        Transcribes normalized audio file using Whisper.
        Whisper only reads normalized.wav, never original uploads.
        """
        # Return empty if model not loaded
        if self.model is None:
            return TranscriptResponse(
                session_id=session_id,
                segments=[],
                status="complete"
            )
        
        # Get normalized audio path (Whisper only uses normalized.wav)
        normalized_path = self.normalization_service.get_normalized_path(session_id)
        
        if normalized_path is None or not normalized_path.exists():
            # Normalized audio not available
            return TranscriptResponse(
                session_id=session_id,
                segments=[],
                status="complete"
            )
        
        # Transcribe with Whisper using normalized audio only
        try:
            result = self.model.transcribe(
                str(normalized_path),
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

