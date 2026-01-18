"""
Speaker Diarization Service (Phase 2.3)

This module defines the contract for speaker diarization output.
Implementation will be added in Phase 2.3.

PHASE 2.3 DIARIZATION OUTPUT CONTRACT
======================================

Artifact Location:
    storage/audio/{session_id}/diarization.json

Output Schema:
    {
        "speakers": [
            {
                "speaker_id": "SPEAKER_0",
                "segments": [
                    { "start": <float>, "end": <float> }
                ]
            },
            {
                "speaker_id": "SPEAKER_1",
                "segments": [
                    { "start": <float>, "end": <float> }
                ]
            }
        ]
    }

Contract Rules:
    1. speaker_id Format:
       - Synthetic identifier (e.g., "SPEAKER_0", "SPEAKER_1", ...)
       - Session-scoped (not globally unique)
       - Sequential numbering starting from 0
       - No role inference (doctor, patient, etc.)
       - No name inference

    2. Segment Mapping:
       - Every segment from vad_segments.json MUST appear exactly once
       - No VAD segment may be omitted
       - No VAD segment may be duplicated
       - Complete coverage of all VAD segments

    3. Segment Timing:
       - start and end times MUST match vad_segments.json exactly
       - No modification of segment boundaries
       - No splitting or merging of VAD segments
       - Preserve original VAD segment timestamps

    4. Non-Overlapping Constraint:
       - Segments from different speakers MUST NOT overlap
       - Each time point belongs to at most one speaker
       - Gaps between segments are allowed (silence)

    5. Input Dependency:
       - Input: vad_segments.json (Phase 2.2 output)
       - Input: normalized.wav (Phase 2.1 output)
       - Must process all segments from vad_segments.json

    6. Output Stability:
       - Same input → same output (deterministic)
       - Speaker IDs may vary across runs (session-scoped)
       - Segment assignments must be consistent

Example Output:
    {
        "speakers": [
            {
                "speaker_id": "SPEAKER_0",
                "segments": [
                    { "start": 0.5, "end": 3.2 },
                    { "start": 8.1, "end": 12.4 }
                ]
            },
            {
                "speaker_id": "SPEAKER_1",
                "segments": [
                    { "start": 3.2, "end": 8.1 },
                    { "start": 12.4, "end": 15.7 }
                ]
            }
        ]
    }

Phase Boundaries:
    - Phase 2.2 (VAD) produces vad_segments.json
    - Phase 2.3 (Diarization) consumes vad_segments.json, produces diarization.json
    - Phase 2.4+ may consume diarization.json for downstream processing

Implementation Status:
    CONTRACT DEFINED - Implementation pending Phase 2.3
"""

from typing import List, Dict, Any
from pathlib import Path


class DiarizationService:
    """
    Speaker Diarization Service (Phase 2.3).
    
    Contract: See module-level documentation above.
    
    This class will be implemented in Phase 2.3 to assign speaker IDs
    to segments from vad_segments.json.
    
    Phase 2.3 LOCKED CONTRACT:
    - Input: vad_segments.json (from Phase 2.2)
    - Input: normalized.wav (from Phase 2.1)
    - Output: diarization.json (this contract)
    - Output location: storage/audio/{session_id}/diarization.json
    """
    
    def __init__(self):
        """Initialize diarization service."""
        self.storage_dir = Path(__file__).parent.parent / "storage" / "audio"
    
    # Implementation will be added in Phase 2.3
    # Method signatures to be defined:
    # - def diarize(session_id: str, normalized_audio_path: Path) -> Optional[Path]
    # - def get_diarization_path(session_id: str) -> Optional[Path]
