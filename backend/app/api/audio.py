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


@router.get("/{session_id}/diarization")
async def get_diarization(session_id: str):
    """
    Get speaker diarization results for a session.
    
    Returns speaker segments with speaker IDs (SPEAKER_0, SPEAKER_1, etc.)
    showing which parts of the audio belong to which speaker.
    
    The diarization is automatically generated when audio is uploaded.
    """
    import json
    
    # Validate session_id format
    if len(session_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    # Check if diarization file exists
    diarization_path = STORAGE_DIR / session_id / "diarization.json"
    
    if not diarization_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Diarization not found. Please upload audio first or wait for processing to complete."
        )
    
    # Load and return diarization data
    try:
        with open(diarization_path, 'r') as f:
            diarization_data = json.load(f)
        return diarization_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load diarization: {str(e)}")


@router.get("/{session_id}/vad")
async def get_vad_segments(session_id: str):
    """
    Get Voice Activity Detection (VAD) segments for a session.
    
    Returns segments where speech was detected (non-silence regions).
    This is useful for debugging and understanding the diarization input.
    """
    import json
    
    # Validate session_id format
    if len(session_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    # Check if VAD file exists
    vad_path = STORAGE_DIR / session_id / "vad_segments.json"
    
    if not vad_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="VAD segments not found. Please upload audio first or wait for processing to complete."
        )
    
    # Load and return VAD data
    try:
        with open(vad_path, 'r') as f:
            vad_data = json.load(f)
        return vad_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load VAD segments: {str(e)}")

