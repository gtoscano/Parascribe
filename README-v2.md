# Parascribe v2: session voiceprints

V2 adds incremental transcription and stable, anonymous speaker recognition
within a conversation. The existing WhisperX `/v1` batch API is unchanged;
v2 sentence/realtime transcription uses multilingual **Parakeet TDT 0.6B v3
INT8 on ONNX Runtime**, the efficient path already proven in Voikoro.

## What “speaker recognition” means here

WhisperX diarization labels (`SPEAKER_00`, `SPEAKER_01`) are local to one audio
file. V2 additionally extracts a normalized speaker embedding, compares it to
the embeddings already enrolled in the same session, and returns a stable id
such as `speaker_1`.

There is intentionally no global speaker directory. Two sessions can each
contain `speaker_1`, but Parascribe never compares those voiceprints. A sentence
upload must contain one speaker; use `/v1/transcribe` for a multi-speaker file.

The default similarity thresholds are starting points, not universal identity
guarantees. Calibrate them with representative microphones, rooms, languages,
and speakers before using the result for anything consequential. Do not use
voiceprints as authentication.

## Storage and deletion

Session transcripts and voiceprints are stored in `/data/sessions.sqlite3`, a
volume shared by the replicas. Uploaded v2 audio is deleted after processing;
the transcript, anonymous speaker id, job result, and timings remain.

Voiceprints are biometric data. They are held in a separate table so the client
can delete the vectors while retaining the transcript:

```http
DELETE /v2/sessions/{session_id}/voiceprints
```

Deletion also disables future voiceprint capture for that session. Ending a
session can do both operations atomically from the client's point of view:

```json
POST /v2/sessions/{session_id}/end
{"delete_voiceprints": true}
```

SQLite secure deletion is enabled and its write-ahead log is truncated after
voiceprint deletion. Operators must separately apply the same retention policy
to backups, filesystem snapshots, logs, and exported database copies.

## Sentence API

Create a session:

```bash
curl -s http://localhost:8357/v2/sessions \
  -H 'content-type: application/json' \
  -d '{"capture_voiceprints":true,"metadata":{"conversation_id":"demo"}}'
```

Submit a single-speaker sentence or utterance. This returns `202` and a job id,
using the same queue and replica-prefixed job routing as v1:

```bash
curl -s http://localhost:8357/v2/sessions/$SESSION_ID/sentences \
  -F file=@sentence.wav -F language=en

curl -s http://localhost:8357/v1/jobs/$JOB_ID
```

The completed result contains the transcript and session-local identity:

```json
{
  "text": "I agree with the proposal.",
  "speaker": {
    "id": "speaker_1",
    "similarity": 0.83,
    "enrolled": false,
    "signature_stored": true
  }
}
```

On a speaker's first utterance, `enrolled` is `true` and `similarity` is `null`.
If voiceprint capture was disabled or deleted, `id` is `null` while
transcription continues normally.

Fetch the durable session transcript at any time:

```http
GET /v2/sessions/{session_id}
```

## Realtime WebSocket API

Connect to:

```text
ws://localhost:8357/v2/sessions/{session_id}/stream
```

The v2 stream is sentence-realtime: audio travels continuously over one
connection, and each committed utterance is transcribed and identified through
the worker queue. Parakeet transcription runs on CPU; only the small WeSpeaker
voice embedding uses the configured Torch device. It does not yet emit unstable
token-by-token partial text.

Protocol:

1. Server sends `{"type":"ready", ...}`.
2. Client sends `{"type":"start","format":"pcm_s16le","sample_rate":16000,"channels":1}`.
3. Client sends binary mono PCM16 audio frames.
4. Client sends `{"type":"commit","utterance_id":"client-1"}` at a sentence or VAD boundary.
5. Server sends `utterance_queued`, then `utterance` with the final result.
6. Repeat steps 3–5 while prior utterances are processed; up to eight committed
   utterances may be buffered per connection.
7. Client sends `{"type":"end","delete_voiceprints":true}` and receives
   `session_ended`.

The client may also send `{"type":"delete_voiceprints"}` before the end or
`{"type":"ping"}`. One committed utterance is limited to 60 seconds by
default. A disconnect does not end the session, allowing reconnection.

Nginx is configured to forward WebSocket upgrade headers. Once a connection is
established it remains on one replica, while the SQLite session remains visible
to every replica.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `SESSION_ASR_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | V2 sentence/realtime ASR |
| `SESSION_ASR_MODEL_DIR` | empty | Optional local ONNX model directory; otherwise onnx-asr uses its cache/downloader |
| `SESSION_ASR_QUANTIZATION` | `int8` | Parakeet ONNX weights |
| `SESSION_ASR_THREADS` | half the CPUs | ONNX Runtime intra-op threads |
| `SESSION_ASR_MIN_CONFIDENCE` | `0` | Optional mean token-confidence floor |
| `VOICEPRINT_MODEL` | `pyannote/wespeaker-voxceleb-resnet34-LM` | Speaker embedding model |
| `VOICEPRINT_SIMILARITY_THRESHOLD` | `0.50` | Minimum cosine similarity for a match |
| `VOICEPRINT_UPDATE_THRESHOLD` | `0.70` | Minimum similarity before updating a speaker centroid |
| `STREAM_MAX_UTTERANCE_SECONDS` | `60` | WebSocket commit size limit |
| `SENTENCE_MAX_UPLOAD_BYTES` | `33554432` | REST sentence upload limit |

The centroid is updated only on confident matches to reduce identity drift.
The current implementation does not resolve overlapping speakers inside one
sentence, merge accidentally split identities, authenticate clients, or encrypt
the database at rest. Those are the next production-hardening items.
