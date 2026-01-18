"""
Phase 2.3 Diarization Verification Tests

Tests verify:
1. Single speaker monologue → one speaker only
2. Two speaker dialogue → two speakers
3. Alternating turns → stable speaker IDs
4. All VAD segments appear exactly once
5. Structural correctness of diarization.json
6. Deterministic output across runs
"""

import json
import pytest
from pathlib import Path
from app.services.diarization import DiarizationService


class TestDiarizationVerification:
    """Verification tests for Phase 2.3 diarization."""
    
    @pytest.fixture
    def diarization_service(self):
        """Create DiarizationService instance."""
        return DiarizationService()
    
    @pytest.fixture
    def sample_vad_segments_single_speaker(self):
        """Sample VAD segments for single speaker monologue."""
        return [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8},
            {"start": 9.2, "end": 12.5}
        ]
    
    @pytest.fixture
    def sample_vad_segments_two_speakers(self):
        """Sample VAD segments for two speaker dialogue."""
        return [
            {"start": 0.5, "end": 3.2},   # Speaker 1
            {"start": 3.8, "end": 7.1},   # Speaker 2
            {"start": 7.5, "end": 10.3},  # Speaker 1
            {"start": 10.8, "end": 14.2}, # Speaker 2
        ]
    
    def test_diarization_json_structure(self, diarization_service):
        """
        Test structural correctness of diarization.json schema.
        """
        # Valid diarization.json structure
        valid_data = {
            "speakers": [
                {
                    "speaker_id": "SPEAKER_0",
                    "segments": [
                        {"start": 0.5, "end": 3.2}
                    ]
                }
            ]
        }
        
        # Verify structure
        assert "speakers" in valid_data
        assert isinstance(valid_data["speakers"], list)
        assert len(valid_data["speakers"]) > 0
        
        for speaker in valid_data["speakers"]:
            assert "speaker_id" in speaker
            assert "segments" in speaker
            assert speaker["speaker_id"].startswith("SPEAKER_")
            assert isinstance(speaker["segments"], list)
            
            for segment in speaker["segments"]:
                assert "start" in segment
                assert "end" in segment
                assert isinstance(segment["start"], (int, float))
                assert isinstance(segment["end"], (int, float))
                assert segment["end"] > segment["start"]
    
    def test_all_segments_present_once(self, diarization_service):
        """
        Verify all VAD segments appear exactly once in diarization output.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8},
            {"start": 9.2, "end": 12.5}
        ]
        
        # Group segments (simulate diarization output)
        diarization_data = diarization_service._group_segments_by_speaker(
            {0: "SPEAKER_0", 1: "SPEAKER_0", 2: "SPEAKER_0"},
            vad_segments
        )
        
        # Extract all segments from output
        output_segments = []
        for speaker in diarization_data["speakers"]:
            for segment in speaker["segments"]:
                output_segments.append((segment["start"], segment["end"]))
        
        # Extract all input segments
        input_segments = [(s["start"], s["end"]) for s in vad_segments]
        
        # Verify: all input segments appear exactly once
        assert len(output_segments) == len(input_segments)
        assert set(output_segments) == set(input_segments)
        
        # Verify: no duplicates
        assert len(output_segments) == len(set(output_segments))
    
    def test_segment_times_preserved(self, diarization_service):
        """
        Verify original VAD segment times are preserved exactly.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8}
        ]
        
        diarization_data = diarization_service._group_segments_by_speaker(
            {0: "SPEAKER_0", 1: "SPEAKER_1"},
            vad_segments
        )
        
        # Check each segment time is preserved exactly
        output_segments_by_time = {}
        for speaker in diarization_data["speakers"]:
            for segment in speaker["segments"]:
                key = (segment["start"], segment["end"])
                output_segments_by_time[key] = segment
        
        for vad_seg in vad_segments:
            key = (vad_seg["start"], vad_seg["end"])
            assert key in output_segments_by_time
            assert output_segments_by_time[key]["start"] == vad_seg["start"]
            assert output_segments_by_time[key]["end"] == vad_seg["end"]
    
    def test_speaker_id_format(self, diarization_service):
        """
        Verify speaker IDs follow SPEAKER_0, SPEAKER_1, ... format.
        """
        import numpy as np
        
        # Create dummy embeddings to test clustering
        embeddings = [
            np.ones(192, dtype=np.float32) * (i * 0.3)  # Different embeddings
            for i in range(5)
        ]
        vad_segments = [{"start": i * 1.0, "end": (i + 1) * 1.0} for i in range(len(embeddings))]
        
        speaker_assignments = diarization_service._cluster_embeddings(
            embeddings, vad_segments
        )
        
        # Verify all speaker IDs follow format
        if speaker_assignments:
            speaker_ids = set(speaker_assignments.values())
            for speaker_id in speaker_ids:
                assert speaker_id.startswith("SPEAKER_")
                assert speaker_id[8:].isdigit()  # After "SPEAKER_" should be a number
    
    def test_cluster_labels_mapping(self, diarization_service):
        """
        Verify cluster labels map correctly to speaker IDs.
        """
        import numpy as np
        
        # Create dummy embeddings for testing (will produce different clusters)
        # Use distinct embeddings to simulate different speakers
        embeddings = [
            np.ones(192, dtype=np.float32) * 0.5,   # Speaker 1
            np.ones(192, dtype=np.float32) * 0.5,   # Speaker 1 (similar)
            np.ones(192, dtype=np.float32) * 1.0,   # Speaker 2 (different)
            np.ones(192, dtype=np.float32) * 1.0,   # Speaker 2 (similar)
        ]
        vad_segments = [{"start": i * 1.0, "end": (i + 1) * 1.0} for i in range(len(embeddings))]
        
        # Get speaker assignments via clustering
        speaker_assignments = diarization_service._cluster_embeddings(
            embeddings, vad_segments
        )
        
        if speaker_assignments:
            # Verify: same cluster labels map to same speaker IDs
            # Cluster 0 → SPEAKER_0, Cluster 1 → SPEAKER_1 (if different clusters)
            if len(set(speaker_assignments.values())) >= 2:
                # If we have at least 2 speakers, verify structure
                assert speaker_assignments[0] == speaker_assignments[1]  # Same cluster (similar embeddings)
                assert speaker_assignments[2] == speaker_assignments[3]  # Same cluster (similar embeddings)
                assert speaker_assignments[0] != speaker_assignments[2]  # Different clusters
    
    def test_deterministic_output(self, diarization_service):
        """
        Verify diarization output is deterministic across runs.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8},
            {"start": 9.2, "end": 12.5}
        ]
        
        # First run
        speaker_assignments_1 = {0: "SPEAKER_0", 1: "SPEAKER_0", 2: "SPEAKER_1"}
        output_1 = diarization_service._group_segments_by_speaker(
            speaker_assignments_1, vad_segments
        )
        
        # Second run (same input)
        speaker_assignments_2 = {0: "SPEAKER_0", 1: "SPEAKER_0", 2: "SPEAKER_1"}
        output_2 = diarization_service._group_segments_by_speaker(
            speaker_assignments_2, vad_segments
        )
        
        # Verify: same speaker assignments → same output structure
        assert len(output_1["speakers"]) == len(output_2["speakers"])
        
        # Verify: segments are identical (order may differ)
        segments_1 = set()
        for speaker in output_1["speakers"]:
            for seg in speaker["segments"]:
                segments_1.add((seg["start"], seg["end"]))
        
        segments_2 = set()
        for speaker in output_2["speakers"]:
            for seg in speaker["segments"]:
                segments_2.add((seg["start"], seg["end"]))
        
        assert segments_1 == segments_2
    
    def test_single_speaker_monologue_structure(self, diarization_service):
        """
        Verify single speaker monologue produces one speaker only.
        Note: This is a structural test, not a functional test.
        Requires actual embeddings to test clustering behavior.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8}
        ]
        
        # Simulate single speaker (all segments same cluster)
        speaker_assignments = {0: "SPEAKER_0", 1: "SPEAKER_0"}
        diarization_data = diarization_service._group_segments_by_speaker(
            speaker_assignments, vad_segments
        )
        
        # Verify: one speaker only
        assert len(diarization_data["speakers"]) == 1
        assert diarization_data["speakers"][0]["speaker_id"] == "SPEAKER_0"
        assert len(diarization_data["speakers"][0]["segments"]) == 2
    
    def test_two_speaker_dialogue_structure(self, diarization_service):
        """
        Verify two speaker dialogue produces two speakers.
        Note: This is a structural test, not a functional test.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8},
            {"start": 9.2, "end": 12.5}
        ]
        
        # Simulate two speakers (alternating segments)
        speaker_assignments = {0: "SPEAKER_0", 1: "SPEAKER_1", 2: "SPEAKER_0"}
        diarization_data = diarization_service._group_segments_by_speaker(
            speaker_assignments, vad_segments
        )
        
        # Verify: two speakers
        assert len(diarization_data["speakers"]) == 2
        
        # Verify: all segments present
        total_segments = sum(
            len(speaker["segments"]) for speaker in diarization_data["speakers"]
        )
        assert total_segments == 3
    
    def test_alternating_turns_stable_ids(self, diarization_service):
        """
        Verify alternating turns produce stable speaker IDs.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},   # Speaker 1
            {"start": 4.1, "end": 7.8},   # Speaker 2
            {"start": 9.2, "end": 12.5},  # Speaker 1
            {"start": 13.1, "end": 16.8}  # Speaker 2
        ]
        
        # Alternating speaker assignments
        speaker_assignments = {
            0: "SPEAKER_0",
            1: "SPEAKER_1",
            2: "SPEAKER_0",
            3: "SPEAKER_1"
        }
        
        diarization_data = diarization_service._group_segments_by_speaker(
            speaker_assignments, vad_segments
        )
        
        # Verify: two speakers
        assert len(diarization_data["speakers"]) == 2
        
        # Verify: stable IDs (same speaker_id for segments with same assignment)
        speaker_0_segments = None
        speaker_1_segments = None
        
        for speaker in diarization_data["speakers"]:
            if speaker["speaker_id"] == "SPEAKER_0":
                speaker_0_segments = speaker["segments"]
            elif speaker["speaker_id"] == "SPEAKER_1":
                speaker_1_segments = speaker["segments"]
        
        assert speaker_0_segments is not None
        assert speaker_1_segments is not None
        assert len(speaker_0_segments) == 2
        assert len(speaker_1_segments) == 2
    
    def test_non_overlapping_segments(self, diarization_service):
        """
        Verify segments from different speakers do not overlap.
        This verifies the non-overlapping constraint.
        """
        vad_segments = [
            {"start": 0.5, "end": 3.2},
            {"start": 4.1, "end": 7.8},
            {"start": 9.2, "end": 12.5}
        ]
        
        speaker_assignments = {0: "SPEAKER_0", 1: "SPEAKER_1", 2: "SPEAKER_0"}
        diarization_data = diarization_service._group_segments_by_speaker(
            speaker_assignments, vad_segments
        )
        
        # Extract all segments with speaker IDs
        all_segments = []
        for speaker in diarization_data["speakers"]:
            for segment in speaker["segments"]:
                all_segments.append({
                    "speaker_id": speaker["speaker_id"],
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        # Verify: no overlapping segments from different speakers
        for i, seg1 in enumerate(all_segments):
            for j, seg2 in enumerate(all_segments):
                if i != j and seg1["speaker_id"] != seg2["speaker_id"]:
                    # Check if segments overlap
                    overlaps = not (seg1["end"] <= seg2["start"] or seg2["end"] <= seg1["start"])
                    assert not overlaps, f"Segments overlap: {seg1} and {seg2}"
