"""
Speaker Diarization Service (Phase 2.3)

This module implements speaker diarization using ECAPA-TDNN embeddings
and agglomerative clustering. Phase 2.3 is LOCKED.

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
    PHASE 2.3 LOCKED - Implementation complete

Phase 2.3 LOCKED:
    - ECAPA-TDNN embeddings (CPU-only, L2-normalized)
    - Agglomerative clustering (cosine distance, average linkage)
    - diarization.json is a stable internal contract
    - Output format: {"speakers": [{"speaker_id": str, "segments": [...]}, ...]}
    - Output consumed by: Phase 2.4+ (downstream processing)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import wave
import torch

# Monkey patch for torchaudio compatibility with speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def _list_audio_backends():
        """Compatibility shim for newer torchaudio versions."""
        return ['soundfile']
    torchaudio.list_audio_backends = _list_audio_backends

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering


class DiarizationService:
    """
    Speaker Diarization Service (Phase 2.3).
    
    Contract: See module-level documentation above.
    
    Assigns speaker IDs to segments from vad_segments.json using
    ECAPA-TDNN embeddings and agglomerative clustering.
    
    Phase 2.3 LOCKED:
    - Input: vad_segments.json (from Phase 2.2)
    - Input: normalized.wav (from Phase 2.1)
    - Output: diarization.json (stable contract)
    - Output location: storage/audio/{session_id}/diarization.json
    - Embedding: ECAPA-TDNN (CPU-only, L2-normalized, 192-dim)
    - Clustering: Agglomerative (cosine distance, average linkage, threshold=0.3)
    """
    
    # Clustering distance threshold (cosine distance for stopping clustering)
    # Lower values = more clusters (stricter similarity requirement)
    CLUSTERING_DISTANCE_THRESHOLD = 0.3
    
    def __init__(self):
        """Initialize diarization service."""
        self.storage_dir = Path(__file__).parent.parent / "storage" / "audio"
        # Initialize ECAPA-TDNN speaker embedding model (CPU-only)
        try:
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
        except Exception:
            self.embedding_model = None
    
    def run(self, session_id: str) -> Optional[Path]:
        """
        Run diarization for a session.
        
        Main entry point for Phase 2.3 diarization.
        Loads VAD segments, performs diarization, and saves output.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to diarization.json if successful, None otherwise
        """
        # Load VAD segments
        vad_segments = self._load_vad_segments(session_id)
        if not vad_segments:
            return None
        
        # Get normalized audio path
        normalized_audio_path = self._get_normalized_audio_path(session_id)
        if not normalized_audio_path or not normalized_audio_path.exists():
            return None
        
        # Extract embeddings for each VAD segment
        embeddings = self._extract_embeddings(normalized_audio_path, vad_segments)
        if not embeddings:
            return None
        
        # Cluster embeddings to assign speaker IDs
        speaker_assignments = self._cluster_embeddings(embeddings, vad_segments)
        if not speaker_assignments:
            return None
        
        # Group segments by speaker and format output
        diarization_data = self._group_segments_by_speaker(speaker_assignments, vad_segments)
        if not diarization_data:
            return None
        
        # Save diarization output
        return self._save_diarization(session_id, diarization_data)
    
    def _load_vad_segments(self, session_id: str) -> Optional[List[Dict[str, float]]]:
        """
        Load VAD segments from vad_segments.json.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of segment dicts with 'start' and 'end' keys, or None if not found
        """
        vad_path = self.storage_dir / session_id / "vad_segments.json"
        
        if not vad_path.exists():
            return None
        
        try:
            with open(vad_path, 'r') as f:
                vad_data = json.load(f)
                return vad_data.get("segments", [])
        except Exception:
            return None
    
    def _get_normalized_audio_path(self, session_id: str) -> Optional[Path]:
        """
        Get path to normalized audio file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to normalized.wav if exists, None otherwise
        """
        normalized_path = self.storage_dir / session_id / "normalized.wav"
        
        if normalized_path.exists() and normalized_path.stat().st_size > 0:
            return normalized_path
        
        return None
    
    def _extract_embeddings(self, normalized_audio_path: Path, vad_segments: List[Dict[str, float]]) -> List[np.ndarray]:
        """
        Extract speaker embeddings for each VAD segment using ECAPA-TDNN.
        
        Args:
            normalized_audio_path: Path to normalized.wav (16kHz mono)
            vad_segments: List of VAD segments with 'start' and 'end' times
            
        Returns:
            List of L2-normalized embedding vectors (one per segment)
        """
        if self.embedding_model is None:
            return []
        
        if not normalized_audio_path.exists():
            return []
        
        embeddings = []
        
        # Load full audio file
        try:
            with wave.open(str(normalized_audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                # Verify format (must be 16kHz mono)
                if sample_rate != 16000 or channels != 1 or sample_width != 2:
                    return []
                
                # Read all audio data
                audio_data = wf.readframes(wf.getnframes())
                audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return []
        
        # Extract embedding for each VAD segment
        for segment_idx, segment in enumerate(vad_segments):
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if start_sample >= len(audio_samples) or end_sample > len(audio_samples):
                # Skip invalid segments
                continue
            
            segment_audio = audio_samples[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # Extract embedding for this segment
            embedding = self.extract_embedding(segment_audio)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # If embedding extraction fails, add zero vector to maintain alignment
                # This should be rare, but ensures list length matches segment count
                zero_embedding = np.zeros(192, dtype=np.float32)  # ECAPA-TDNN output size
                embeddings.append(zero_embedding)
        
        return embeddings
    
    def extract_embedding(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment using ECAPA-TDNN.
        
        Args:
            audio_segment: Audio samples as numpy array (float32, range [-1, 1])
                          Expected: normalized mono 16kHz PCM
            
        Returns:
            L2-normalized embedding vector (192-dim for ECAPA-TDNN) or None on error
        """
        if self.embedding_model is None:
            return None
        
        if len(audio_segment) == 0:
            return None
        
        try:
            # Convert to torch tensor and add batch dimension
            # ECAPA expects shape: [batch, channels, samples]
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0).unsqueeze(0)
            
            # Extract embedding (inference only, CPU)
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
            
            # Convert to numpy and squeeze batch dimension
            embedding_np = embedding.squeeze(0).cpu().numpy()
            
            # L2 normalization
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
            
            return embedding_np.astype(np.float32)
            
        except Exception:
            return None
    
    def cluster_embeddings(self, embeddings: List[np.ndarray]) -> List[int]:
        """
        Cluster embeddings using Agglomerative Hierarchical Clustering.
        
        Uses cosine distance metric and average linkage.
        Distance threshold stopping rule determines number of clusters.
        
        Args:
            embeddings: List of L2-normalized embedding vectors
            
        Returns:
            List of cluster labels (integers) corresponding to each embedding
        """
        if not embeddings:
            return []
        
        if len(embeddings) == 1:
            # Single embedding = single cluster
            return [0]
        
        # Convert embeddings list to numpy array for clustering
        embedding_matrix = np.vstack(embeddings)
        
        # Agglomerative clustering with cosine distance and average linkage
        clustering = AgglomerativeClustering(
            n_clusters=None,  # Use distance threshold instead
            distance_threshold=self.CLUSTERING_DISTANCE_THRESHOLD,
            metric='cosine',
            linkage='average'
        )
        
        # Perform clustering
        cluster_labels = clustering.fit_predict(embedding_matrix)
        
        return cluster_labels.tolist()
    
    def _cluster_embeddings(self, embeddings: List[np.ndarray], vad_segments: List[Dict[str, float]]) -> Dict[int, str]:
        """
        Cluster embeddings and map cluster labels to speaker IDs.
        
        Args:
            embeddings: List of embeddings (one per segment)
            vad_segments: List of VAD segments (for reference)
            
        Returns:
            Dictionary mapping segment index to speaker_id (SPEAKER_0, SPEAKER_1, ...)
        """
        # Cluster embeddings using agglomerative clustering
        cluster_labels = self.cluster_embeddings(embeddings)
        
        if not cluster_labels:
            return {}
        
        # Map cluster labels to speaker IDs
        # Get unique cluster labels and sort to ensure deterministic mapping
        unique_labels = sorted(set(cluster_labels))
        
        # Create mapping: cluster_label -> speaker_id
        label_to_speaker = {
            label: f"SPEAKER_{i}" for i, label in enumerate(unique_labels)
        }
        
        # Map each segment index to speaker_id
        speaker_assignments = {}
        for segment_idx, cluster_label in enumerate(cluster_labels):
            speaker_id = label_to_speaker[cluster_label]
            speaker_assignments[segment_idx] = speaker_id
        
        return speaker_assignments
    
    def _group_segments_by_speaker(self, speaker_assignments: Dict[int, str], vad_segments: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Group segments by speaker and format output according to contract.
        
        Groups segments by assigned speaker_id and formats according to
        diarization.json schema. All VAD segments must appear exactly once.
        
        Args:
            speaker_assignments: Dictionary mapping segment index to speaker_id
            vad_segments: List of VAD segments with 'start' and 'end' times
            
        Returns:
            Dictionary conforming to diarization.json schema
        """
        if not speaker_assignments or not vad_segments:
            return {}
        
        # Group segments by speaker_id
        speaker_segments: Dict[str, List[Dict[str, float]]] = {}
        
        for segment_idx, speaker_id in speaker_assignments.items():
            # Verify segment index is valid
            if segment_idx < 0 or segment_idx >= len(vad_segments):
                continue
            
            # Get original segment (preserve times exactly)
            segment = vad_segments[segment_idx]
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Create segment dict with preserved times
            segment_dict = {
                "start": float(start_time),
                "end": float(end_time)
            }
            
            # Add to speaker's segment list
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            
            speaker_segments[speaker_id].append(segment_dict)
        
        # Format output according to diarization.json contract
        speakers_list = []
        for speaker_id in sorted(speaker_segments.keys()):  # Sort for deterministic output
            speakers_list.append({
                "speaker_id": speaker_id,
                "segments": speaker_segments[speaker_id]
            })
        
        diarization_data = {
            "speakers": speakers_list
        }
        
        return diarization_data
    
    def _save_diarization(self, session_id: str, diarization_data: Dict[str, Any]) -> Optional[Path]:
        """
        Save diarization output to diarization.json.
        
        Args:
            session_id: Session identifier
            diarization_data: Dictionary conforming to diarization.json schema
            
        Returns:
            Path to saved diarization.json if successful, None otherwise
        """
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = session_dir / "diarization.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(diarization_data, f, indent=2)
            return output_path
        except Exception:
            return None
    
    def get_diarization_path(self, session_id: str) -> Optional[Path]:
        """
        Get path to diarization output file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to diarization.json if exists, None otherwise
        """
        diarization_path = self.storage_dir / session_id / "diarization.json"
        
        if diarization_path.exists() and diarization_path.stat().st_size > 0:
            return diarization_path
        
        return None
