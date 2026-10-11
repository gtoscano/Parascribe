# Parascribe v2: session voiceprints

V2 adds incremental transcription and stable, anonymous speaker recognition
within a conversation. The existing WhisperX `/v1` batch API is unchanged;
v2 sentence/realtime transcription uses multilingual **Parakeet TDT 0.6B v3
INT8 on ONNX Runtime** by default. **Cohere Transcribe 03-2026** is also
available locally through `transcribe.cpp`, as in Voikoro. Select either engine
or run both for a later benchmark.

The shared service uses generic Parascribe names: `parascribe/`,
`parascribe_cluster.py`, and the `parascribe-api:local` Docker image.
It installs and imports WhisperX because `/v1` uses it. `/v2` transcription calls Parakeet
and/or Cohere; WeSpeaker handles its voice signatures separately. V2 job polling
also uses the existing `/v1/jobs` routes, which are shared job infrastructure,
not a choice of transcription model.

## Browser tester

After starting the service, open **http://localhost:8357/v2/test**. Parascribe
serves the HTML and JavaScript itself; no separate frontend server or desktop
app is required.

```bash
docker build -t parascribe-api:local ./parascribe
python3 parascribe_cluster.py up --replicas 1
```

1. Create a session with speaker recognition enabled, or paste an existing
   session ID and open it. Session IDs are shown on the page for reconnection.
2. Choose Parakeet, Cohere, or Compare both. Select English or Spanish (or
   another listed language); Cohere requires a language and configured weights.
3. Upload a single-speaker audio file, or start the microphone, speak, and click
   **Send sentence**. The microphone stays on. **Stop microphone** submits the
   remaining audio. A 55-second automatic boundary prevents oversized commits.
4. Read the transcript, session speaker ID, voice similarity, and—in comparison
   mode—both transcripts and inference timings. These are observations, not an
   accuracy benchmark. For voice matching, try several seconds of clear speech
   from each person, one person at a time.
5. Delete signatures, or end the session with **Delete voice signatures when
   ending** selected (the tester's default). Transcripts are retained. Download
   results exports session history plus the model comparisons collected by the
   current page; reloading recovers primary transcripts, not past comparisons.

The microphone requires HTTPS or a localhost origin. If Parascribe runs on a
remote machine, use your HTTPS reverse proxy or forward its port:

```bash
ssh -L 8357:localhost:8357 your-user@your-server
```

Then open the localhost URL in your own browser. HTTP over a LAN IP can be used
for file uploads, but browsers generally block microphone capture there. Serve
the tester through Parascribe rather than opening the HTML as `file://`; the
same-origin API and WebSocket requests avoid CORS configuration.

The tester converts microphone input to mono 16 kHz PCM16 with an AudioWorklet.
Sentence boundaries are manual; automatic speech/silence detection remains
future work. Closing the page stops microphone capture and disconnects, but
does not end the session or delete its signatures: use the explicit controls.

## Model modes

Both REST sentence uploads and WebSocket streams accept `asr_mode`:

| Mode | Session transcript | Additional output |
|---|---|---|
| `parakeet` (default) | Parakeet | Parakeet result and timing |
| `cohere` | Cohere | Cohere result and timing |
| `compare` | Parakeet | Both independent transcripts and timings |

For `cohere` and `compare`, specify `language` (for example, `en` or `es`).
No text-based language guess or Parakeet noise filter precedes Cohere, so both
engines receive identical decoded audio in comparison mode. Models are lazy
loaded per worker. Comparison runs sequentially and takes longer than a single
engine request; its total latency is not either engine's standalone latency.

REST example after creating a session:

```bash
curl -s http://localhost:8357/v2/sessions/$SESSION_ID/sentences \
  -F file=@sentence.wav -F language=es -F asr_mode=compare
```

WebSocket start example:

```json
{"type":"start","asr_mode":"compare","language":"es","sample_rate":16000,"channels":1}
```

Results include `transcription.mode`, `transcription.primary`, and
`transcription.outputs.parakeet` / `transcription.outputs.cohere`. Each output
contains text, language, model, quantization, device, inference seconds, and
real-time factor (inference seconds divided by audio seconds; lower is faster).
The separate `prepare_*_model_s` timings show model loading overhead.
These outputs persist in each job's `result.json`; the session transcript stores
only the primary text. If the primary model produces no text, the job reports an
error but its model outputs remain in `result.json` for comparison. There is no
silent fallback if an engine fails. Voiceprint extraction still runs only once
per successful utterance and is independent of the selected ASR engine.

No accuracy claim is inferred from differences between transcripts. A later
benchmark needs retained source recordings with human reference transcripts;
ordinary v2 requests continue to delete their uploaded audio. Test the deployed
INT8/Q8 models using WER, names/numbers, omissions, and warm sentence latency,
with English and Spanish scored separately. Model outputs do not include
word timestamps; Cohere text is not aligned to Parakeet words.

### Local Cohere setup

The Docker requirements install `transcribe-cpp==0.2.4`, matching Voikoro's lock.
Download the same pinned Q8 model as Voikoro before selecting Cohere:

```bash
python3 fetch_cohere.py
```

The downloader writes `models/cohere-transcribe-03-2026-Q8_0.gguf` (about 2.4 GB).
Generated compose mounts `./models` read-only at `/models`. Rebuild the image
to install the runtime. Cohere/compare requests return 503 until the model file
is configured; Parakeet can be used without downloading Cohere.

`COHERE_BACKEND=cpu` is the reproducible default. For GPU inference, provide a
CUDA-enabled `libtranscribe` build and set `TRANSCRIBE_LIBRARY` and
`COHERE_BACKEND=auto` in your deployment. Installing the Python package alone
does not establish CUDA support.

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
| `SESSION_ASR_MODE` | `parakeet` | Default mode: `parakeet`, `cohere`, or `compare`; requests may override |
| `COHERE_MODEL_PATH` | empty (compose: `/models/cohere-transcribe-03-2026-Q8_0.gguf`) | Local Cohere GGUF weights |
| `COHERE_BACKEND` | `cpu` | Local transcribe.cpp backend |
| `SESSION_ASR_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | V2 sentence/realtime ASR |
| `SESSION_ASR_MODEL_DIR` | empty | Optional local ONNX model directory; otherwise onnx-asr uses its cache/downloader |
| `SESSION_ASR_QUANTIZATION` | `int8` | Parakeet ONNX weights |
| `SESSION_ASR_THREADS` | half the CPUs | Parakeet ONNX intra-op / Cohere session threads |
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
