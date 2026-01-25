from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from app.models.audio import AudioUploadResponse
from app.services.session_pipeline import process_session

router = APIRouter()

# Ensure storage directory exists
STORAGE_DIR = Path(__file__).parent.parent / "storage" / "audio"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{session_id}/audio", response_model=AudioUploadResponse)
async def upload_audio(session_id: str, file: UploadFile = File(...)):
    """
    Upload audio file for a session.
    Saves the file to storage/audio/ with session_id prefix.
    Triggers the full processing pipeline (normalization, transcription,
    VAD, and diarization) for the session.
    """
    # Validate session_id format (basic UUID check)
    if len(session_id) != 36:  # UUID format check
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    # Generate filename with session_id prefix
    file_extension = Path(file.filename).suffix if file.filename else ".wav"
    filename = f"{session_id}_{file.filename or 'audio'}{file_extension}"
    file_path = STORAGE_DIR / filename
    
    # Save file to disk
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")
    
    # Run the full session processing pipeline once per session
    # (Normalization, transcription, VAD, diarization)
    try:
        process_session(session_id)
    except Exception:
        # Pipeline failures should not crash the upload endpoint
        # Errors will surface during downstream verification
        pass
    
    return AudioUploadResponse(status="accepted")


@router.get("/{session_id}/transcript")
async def get_transcript(session_id: str):
    """
    Get transcript for a session.
    Returns stubbed transcript data for now.
    """
    from app.services.transcription import TranscriptionService
    
    # Validate session_id format
    if len(session_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    transcription_service = TranscriptionService()
    transcript = transcription_service.get_transcript(session_id)
    
    return transcript

