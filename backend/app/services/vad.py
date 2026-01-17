import webrtcvad
import wave
import json
from pathlib import Path
from typing import List, Tuple, Optional


class VoiceActivityDetectionService:
    """
    Voice Activity Detection (VAD) service.
    Uses webrtcvad for conservative speech detection on normalized audio.
    Only processes normalized.wav (16kHz mono).
    """
    
    # Frame duration in milliseconds (10, 20, or 30)
    FRAME_DURATION_MS = 30
    
    # Aggressiveness mode (0-3, higher = more conservative)
    VAD_MODE = 2
    
    # Merge gaps smaller than this (seconds)
    GAP_MERGE_THRESHOLD = 0.3
    
    # Discard segments shorter than this (seconds)
    MIN_SEGMENT_DURATION = 0.5
    
    def __init__(self):
        """Initialize VAD service."""
        self.storage_dir = Path(__file__).parent.parent / "storage" / "audio"
        self.vad = webrtcvad.Vad(self.VAD_MODE)
    
    def detect_speech(self, normalized_audio_path: Path) -> List[Tuple[float, float]]:
        """
        Detect speech segments in normalized audio.
        
        Args:
            normalized_audio_path: Path to normalized.wav (16kHz mono)
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        if not normalized_audio_path.exists():
            return []
        
        try:
            # Read WAV file
            with wave.open(str(normalized_audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                # Verify audio format (must be 16kHz mono 16-bit)
                if sample_rate != 16000 or channels != 1 or sample_width != 2:
                    return []
                
                # Read all audio data
                audio_data = wf.readframes(wf.getnframes())
            
            # Process frames
            # Frame size in samples
            frame_size_samples = int(sample_rate * self.FRAME_DURATION_MS / 1000.0)
            # Frame size in bytes (16-bit = 2 bytes per sample)
            frame_size_bytes = frame_size_samples * sample_width
            num_frames = len(audio_data) // frame_size_bytes
            
            speech_frames = []
            
            # Detect speech in each frame
            for i in range(num_frames):
                frame_start = i * frame_size_bytes
                frame_end = frame_start + frame_size_bytes
                frame = audio_data[frame_start:frame_end]
                
                # VAD requires exactly frame_size_bytes
                if len(frame) != frame_size_bytes:
                    break
                
                is_speech = self.vad.is_speech(frame, sample_rate)
                if is_speech:
                    # Convert frame index to time
                    start_time = i * self.FRAME_DURATION_MS / 1000.0
                    speech_frames.append(start_time)
            
            # Convert speech frames to segments
            segments = self._frames_to_segments(speech_frames)
            
            # Merge small gaps
            segments = self._merge_gaps(segments)
            
            # Filter short segments
            segments = self._filter_short_segments(segments)
            
            return segments
            
        except Exception:
            # Return empty on any error
            return []
    
    def _frames_to_segments(self, speech_frames: List[float]) -> List[Tuple[float, float]]:
        """
        Convert list of speech frame times to continuous segments.
        
        Args:
            speech_frames: List of frame start times where speech was detected
            
        Returns:
            List of (start, end) segment tuples
        """
        if not speech_frames:
            return []
        
        segments = []
        frame_duration = self.FRAME_DURATION_MS / 1000.0
        start_time = speech_frames[0]
        end_time = speech_frames[0] + frame_duration
        
        for i in range(1, len(speech_frames)):
            current_frame = speech_frames[i]
            expected_next = end_time
            
            # If frames are consecutive (within tolerance)
            if abs(current_frame - expected_next) < frame_duration * 1.5:
                # Extend current segment
                end_time = current_frame + frame_duration
            else:
                # Save current segment and start new one
                segments.append((start_time, end_time))
                start_time = current_frame
                end_time = current_frame + frame_duration
        
        # Add final segment
        segments.append((start_time, end_time))
        
        return segments
    
    def _merge_gaps(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge segments with gaps smaller than threshold.
        
        Args:
            segments: List of (start, end) segment tuples
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for i in range(1, len(segments)):
            prev_end = merged[-1][1]
            curr_start = segments[i][0]
            gap = curr_start - prev_end
            
            if gap < self.GAP_MERGE_THRESHOLD:
                # Merge segments
                merged[-1] = (merged[-1][0], segments[i][1])
            else:
                # Keep separate
                merged.append(segments[i])
        
        return merged
    
    def _filter_short_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filter out segments shorter than minimum duration.
        
        Args:
            segments: List of (start, end) segment tuples
            
        Returns:
            Filtered segments
        """
        filtered = []
        for start, end in segments:
            duration = end - start
            if duration >= self.MIN_SEGMENT_DURATION:
                filtered.append((start, end))
        
        return filtered
    
    def save_vad_segments(self, session_id: str, segments: List[Tuple[float, float]]) -> Optional[Path]:
        """
        Save VAD segments to JSON file.
        
        Args:
            session_id: Session identifier
            segments: List of (start, end) speech segments
            
        Returns:
            Path to saved JSON file if successful, None otherwise
        """
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = session_dir / "vad_segments.json"
        
        try:
            # Convert tuples to list of dicts for JSON serialization
            vad_data = {
                "segments": [
                    {"start": float(start), "end": float(end)}
                    for start, end in segments
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(vad_data, f, indent=2)
            
            return output_path
            
        except Exception:
            return None
    
    def process_normalized_audio(self, session_id: str, normalized_audio_path: Path) -> Optional[Path]:
        """
        Process normalized audio and save VAD segments.
        
        Args:
            session_id: Session identifier
            normalized_audio_path: Path to normalized.wav
            
        Returns:
            Path to saved VAD segments JSON if successful, None otherwise
        """
        # Detect speech segments
        segments = self.detect_speech(normalized_audio_path)
        
        # Save segments
        return self.save_vad_segments(session_id, segments)
