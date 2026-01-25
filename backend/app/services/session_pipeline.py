from pathlib import Path
import json
from typing import Optional

from app.services.audio_normalization import AudioNormalizationService
from app.services.transcription import TranscriptionService
from app.services.vad import VoiceActivityDetectionService
from app.services.diarization import DiarizationService


"""
Session processing pipeline (orchestration only).

Phase ordering:
1. Phase 2.1 - Audio normalization
2. Phase 2.2 - Voice activity detection (VAD)
3. Phase 1.x - Transcription (Whisper)
4. Phase 2.3 - Diarization (speaker clustering)

Important:
- This module contains NO signal processing logic.
- It only coordinates existing services.
- Services do NOT call each other directly.
"""


STORAGE_DIR = Path(__file__).parent.parent / "storage" / "audio"


def _find_original_audio_path(session_id: str) -> Optional[Path]:
    """
    Find the original uploaded audio file for a session.

    The upload route stores audio as: {session_id}_{filename}{ext}
    under STORAGE_DIR.
    """
    if not STORAGE_DIR.exists():
        return None

    for file_path in STORAGE_DIR.glob(f"{session_id}_*"):
        if file_path.is_file():
            return file_path

    return None


def normalize_audio(session_id: str) -> Optional[Path]:
    """
    Phase 2.1 - Normalize audio for a session.

    Returns:
        Path to normalized.wav if successful, None otherwise.
    """
    original_path = _find_original_audio_path(session_id)
    if original_path is None:
        return None

    service = AudioNormalizationService()
    return service.normalize_audio(session_id, original_path)


def transcribe_audio(session_id: str) -> Optional[Path]:
    """
    Run transcription for a session and persist transcript.json.

    This uses the existing TranscriptionService and writes the
    returned transcript to:
        storage/audio/{session_id}/transcript.json
    """
    service = TranscriptionService()
    transcript = service.get_transcript(session_id)

    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    output_path = session_dir / "transcript.json"

    try:
        # Pydantic BaseModel supports .dict() for serialization
        data = transcript.dict()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path
    except Exception:
        return None


def run_vad(session_id: str) -> Optional[Path]:
    """
    Phase 2.2 - Run VAD on normalized audio and persist vad_segments.json.

    Uses VoiceActivityDetectionService and expects:
        storage/audio/{session_id}/normalized.wav
    """
    normalized_path = STORAGE_DIR / session_id / "normalized.wav"
    if not normalized_path.exists():
        return None

    vad_service = VoiceActivityDetectionService()
    return vad_service.process_normalized_audio(session_id, normalized_path)


def run_diarization(session_id: str) -> Optional[Path]:
    """
    Phase 2.3 - Run diarization and persist diarization.json.

    Adds temporary verification logging.
    """
    print("DIARIZATION START", session_id)

    diarization_service = DiarizationService()
    output_path = diarization_service.run(session_id)

    print("DIARIZATION COMPLETE", output_path)
    return output_path


def process_session(session_id: str) -> None:
    """
    Orchestrate all processing phases for a session.

    This function is intended to be called once per session, after
    the audio has been uploaded and the session_id is known.
    """
    # Phase 2.1 - Normalization
    normalize_audio(session_id)

    # Phase 1 - Transcription (Whisper)
    transcribe_audio(session_id)

    # Phase 2.2 - VAD
    run_vad(session_id)

    # Phase 2.3 - Diarization
    run_diarization(session_id)

