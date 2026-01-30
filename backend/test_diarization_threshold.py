"""
Test script to experiment with different diarization thresholds.

Usage:
    python test_diarization_threshold.py <session_id> <threshold>

Example:
    python test_diarization_threshold.py 703d8839-81c3-41be-bc5d-8980e6e85601 0.2
"""

import sys
import torchaudio
torchaudio.list_audio_backends = lambda: ['soundfile']

from app.services.diarization import DiarizationService
import json

if len(sys.argv) < 3:
    print("Usage: python test_diarization_threshold.py <session_id> <threshold>")
    print("Example: python test_diarization_threshold.py 703d8839-81c3-41be-bc5d-8980e6e85601 0.2")
    sys.exit(1)

session_id = sys.argv[1]
threshold = float(sys.argv[2])

print(f"\n{'='*60}")
print(f"Testing Diarization with Threshold: {threshold}")
print(f"Session ID: {session_id}")
print(f"{'='*60}\n")

# Create service and override threshold
service = DiarizationService()
service.CLUSTERING_DISTANCE_THRESHOLD = threshold

# Run diarization
result = service.run(session_id)

if result:
    print(f"\n✓ Diarization completed successfully!")
    print(f"Output: {result}\n")
    
    # Read and display the results
    with open(result, 'r') as f:
        diarization_data = json.load(f)
    
    num_speakers = len(diarization_data['speakers'])
    print(f"{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Number of speakers detected: {num_speakers}\n")
    
    for speaker in diarization_data['speakers']:
        speaker_id = speaker['speaker_id']
        num_segments = len(speaker['segments'])
        total_duration = sum(seg['end'] - seg['start'] for seg in speaker['segments'])
        print(f"{speaker_id}:")
        print(f"  - Segments: {num_segments}")
        print(f"  - Total speaking time: {total_duration:.2f}s")
        print(f"  - Segment times: {speaker['segments'][:3]}{'...' if num_segments > 3 else ''}\n")
else:
    print("\n✗ Diarization failed!")
    print("Check that:")
    print("  1. The session ID exists")
    print("  2. normalized.wav and vad_segments.json are present")
    print("  3. The model loaded successfully\n")
