import subprocess
from pathlib import Path
from typing import Optional


class AudioNormalizationService:
    """
    Audio normalization service.
    Converts any audio format to canonical WAV format:
    - sample_rate: 16000 Hz
    - channels: 1 (mono)
    - codec: pcm_s16le
    """
    
    def __init__(self):
        """Initialize normalization service."""
        self.storage_dir = Path(__file__).parent.parent / "storage" / "audio"
    
    def normalize_audio(self, session_id: str, input_file_path: Path) -> Optional[Path]:
        """
        Normalize audio file to canonical format.
        
        Args:
            session_id: Session identifier
            input_file_path: Path to original audio file
            
        Returns:
            Path to normalized.wav if successful, None otherwise
        """
        # Create session directory
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path: storage/audio/{session_id}/normalized.wav
        output_path = session_dir / "normalized.wav"
        
        # Check if input file exists
        if not input_file_path.exists():
            return None
        
        # Build ffmpeg command for deterministic conversion
        # -ar 16000: sample rate 16000 Hz
        # -ac 1: mono channel
        # -acodec pcm_s16le: PCM signed 16-bit little-endian
        # -y: overwrite output file
        # -nostdin: non-interactive mode
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(input_file_path),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            "-nostdin",
            str(output_path)
        ]
        
        try:
            # Run ffmpeg conversion
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check if conversion succeeded
            if result.returncode != 0:
                # ffmpeg failed
                return None
            
            # Verify output file exists and is valid
            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path
            else:
                return None
                
        except Exception:
            # Any exception during normalization
            return None
    
    def get_normalized_path(self, session_id: str) -> Optional[Path]:
        """
        Get path to normalized audio file for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to normalized.wav if exists, None otherwise
        """
        normalized_path = self.storage_dir / session_id / "normalized.wav"
        
        if normalized_path.exists() and normalized_path.stat().st_size > 0:
            return normalized_path
        
        return None
