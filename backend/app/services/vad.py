import webrtcvad
import wave
import json
import numpy as np
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
    
    # Minimum voicing density ratio (voiced_frames / total_frames)
    MIN_VOICING_DENSITY = 0.6
    
    # Minimum relative energy variance (std/mean) for rejecting static/hiss noise
    MIN_RELATIVE_ENERGY_VARIANCE = 0.05
    EPSILON = 1e-6  # Small epsilon to avoid division by zero
    
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
            
            # Track frame-level VAD decisions and RMS energy for filtering
            frame_decisions = []  # List of (frame_time, is_speech) tuples
            frame_energies = []  # List of (frame_time, rms_energy) tuples
            speech_frames = []
            
            # Detect speech in each frame and compute RMS energy
            for i in range(num_frames):
                frame_start = i * frame_size_bytes
                frame_end = frame_start + frame_size_bytes
                frame = audio_data[frame_start:frame_end]
                
                # VAD requires exactly frame_size_bytes
                if len(frame) != frame_size_bytes:
                    break
                
                # Convert frame index to time
                frame_time = i * self.FRAME_DURATION_MS / 1000.0
                is_speech = self.vad.is_speech(frame, sample_rate)
                
                # Compute RMS energy for this frame
                # Convert bytes to 16-bit samples
                frame_samples = np.frombuffer(frame, dtype=np.int16)
                # Normalize int16 samples to float32 in range [-1, 1]
                normalized_samples = frame_samples.astype(np.float32) / 32768.0
                # Compute RMS: sqrt(mean(samples^2))
                rms_energy = np.sqrt(np.mean(normalized_samples ** 2))
                
                # Frame-level debug print
                frame_duration = self.FRAME_DURATION_MS / 1000.0
                print(
                    "FRAME",
                    "t=", round(i * frame_duration, 3),
                    "vad=", is_speech,
                    "rms=", round(float(rms_energy), 6)
                )
                
                # Track all frame decisions and energies for filtering
                frame_decisions.append((frame_time, is_speech))
                frame_energies.append((frame_time, rms_energy))
                
                if is_speech:
                    speech_frames.append(frame_time)
            
            # Convert speech frames to segments
            segments = self._frames_to_segments(speech_frames)
            
            # Merge small gaps
            segments = self._merge_gaps(segments)
            
            # Apply filters in correct order:
            # 1. Minimum duration
            segments = self._filter_short_segments(segments)
            
            # 2. Energy modulation rejection (relative variance)
            segments = self._filter_low_energy_variance(segments, frame_energies)
            
            # 3. Voiced frame ratio
            segments = self._filter_low_voicing_density(segments, frame_decisions, frame_energies)
            
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
    
    def _filter_low_voicing_density(self, segments: List[Tuple[float, float]], frame_decisions: List[Tuple[float, bool]], frame_energies: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filter out segments with low voicing density (static/hiss noise rejection).
        Only segments where at least MIN_VOICING_DENSITY of frames are voiced are kept.
        
        Args:
            segments: List of (start, end) segment tuples
            frame_decisions: List of (frame_time, is_speech) tuples for all frames
            frame_energies: List of (frame_time, rms_energy) tuples for debug output
            
        Returns:
            Filtered segments with sufficient voicing density
        """
        if not segments or not frame_decisions:
            return segments
        
        filtered = []
        frame_duration = self.FRAME_DURATION_MS / 1000.0
        
        for segment_start, segment_end in segments:
            # Count frames within this segment
            total_frames = 0
            voiced_frames = 0
            segment_energies = []
            
            for frame_time, is_speech in frame_decisions:
                # Check if frame is within segment (with small tolerance for frame boundaries)
                if segment_start <= frame_time < segment_end or \
                   (segment_start - frame_duration < frame_time <= segment_end):
                    total_frames += 1
                    if is_speech:
                        voiced_frames += 1
            
            # Collect energy values for debug output
            for frame_time, rms_energy in frame_energies:
                if segment_start <= frame_time < segment_end or \
                   (segment_start - frame_duration < frame_time <= segment_end):
                    segment_energies.append(rms_energy)
            
            # Skip if no frames found (shouldn't happen, but safety check)
            if total_frames == 0:
                continue
            
            # Calculate voicing density
            voicing_density = voiced_frames / total_frames
            
            # Compute energy statistics for debug output
            if len(segment_energies) >= 2:
                energy_mean = np.mean(segment_energies)
                energy_std = np.std(segment_energies)
            else:
                energy_mean = 0.0
                energy_std = 0.0
            
            # Segment-level debug print
            print(
                "DEBUG VAD SEGMENT",
                "start=", round(segment_start, 3),
                "end=", round(segment_end, 3),
                "frames=", total_frames,
                "voiced_frames=", voiced_frames,
                "voiced_ratio=", round(voicing_density, 3),
                "energy_mean=", round(float(energy_mean), 6),
                "energy_std=", round(float(energy_std), 6),
            )
            
            # Keep segment only if voicing density >= threshold
            if voicing_density >= self.MIN_VOICING_DENSITY:
                filtered.append((segment_start, segment_end))
        
        return filtered
    
    def _filter_low_energy_variance(self, segments: List[Tuple[float, float]], frame_energies: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filter out segments with low relative energy variance (static/hiss noise rejection).
        Uses relative variance (std/mean) to detect static noise vs dynamic speech.
        Only segments where relative energy variance >= threshold are kept.
        
        Args:
            segments: List of (start, end) segment tuples
            frame_energies: List of (frame_time, rms_energy) tuples for ALL frames in segment
            
        Returns:
            Filtered segments with sufficient energy modulation
        """
        if not segments or not frame_energies:
            return segments
        
        filtered = []
        frame_duration = self.FRAME_DURATION_MS / 1000.0
        
        for segment_start, segment_end in segments:
            # Collect RMS energy values for ALL frames within this segment window
            # Not just voiced frames - we need to analyze the entire segment
            segment_energies = []
            
            for frame_time, rms_energy in frame_energies:
                # Check if frame is within segment (with small tolerance for frame boundaries)
                if segment_start <= frame_time < segment_end or \
                   (segment_start - frame_duration < frame_time <= segment_end):
                    segment_energies.append(rms_energy)
            
            # Skip if insufficient frames found
            if len(segment_energies) < 2:
                # Need at least 2 frames to compute variance
                continue
            
            # Compute energy statistics
            energy_mean = np.mean(segment_energies)
            energy_std = np.std(segment_energies)
            
            # Compute relative variance (coefficient of variation)
            # This is more robust than absolute std for detecting static noise
            relative_variance = energy_std / (energy_mean + self.EPSILON)
            
            # Keep segment only if relative energy variance >= threshold
            # Static noise has very low relative variance (nearly constant energy)
            # Real speech has higher relative variance (dynamic energy modulation)
            if relative_variance >= self.MIN_RELATIVE_ENERGY_VARIANCE:
                filtered.append((segment_start, segment_end))
        
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
