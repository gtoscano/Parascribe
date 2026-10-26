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
python3 parascribe_cluster.py up --replicas 1 --gpus 1 --gateway-port 8357 --base-port 18360
```

For the complete gpuccino testing procedure, including remote access without
Tailscale configuration changes, see [TO_TEST.md](TO_TEST.md). Replica ports
8360/8361 are occupied by other services on this machine; use 18360 here.

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
ssh -N -L 18357:127.0.0.1:8357 YOUR_USERNAME@gpuccino
```

Replace `YOUR_USERNAME` with your own server account. Each student runs the
command on their own computer and opens **http://localhost:18357/v2/test** in
their own browser, leaving the SSH terminal open. HTTP over a LAN IP can be used
for file uploads, but browsers generally block microphone capture there. Serve
the tester through Parascribe rather than opening the HTML as `file://`; the
same-origin API and WebSocket requests avoid CORS configuration.

The tester converts microphone input to mono 16 kHz PCM16 with an AudioWorklet.
Choose **Manual sentences**, **Automatic sentences on pause**, or **Live partial
text + automatic sentences** in the microphone behavior selector. Automatic
modes use browser-side energy endpointing with adjustable sensitivity and pause
length; they ignore idle silence and cap each utterance at 20 seconds. This is
not neural VAD or overlapping-speaker diarization. Manual mode retains its
55-second limit and Send sentence works in every mode.

Live mode requests a provisional snapshot decode roughly every 1.5 seconds
while speaking. These are revisable whole-utterance hypotheses from the selected
model(s), not native token streaming. They consume additional compute and may
arrive more slowly under load. Each connection allows only one in-flight
preview, rate-limits requests, and invalidates stale drafts on finalization.
Previews reuse background workers but do not create jobs, save audio or
transcripts, or extract/enroll voiceprints. Finalization performs the normal
transcription and speaker match once. If four final sentences are pending, the
browser stops microphone capture so it cannot build an unbounded backlog.

The **Live transcript** tab shows provisional text plus finalized sentences;
**Final & comparison** shows the detailed retained results and model timings.
Speaker badges identify the matched session speaker on final sentences. The
draft explicitly shows **Speaker pending final voice match** rather than
guessing from the previous speaker. Give a session speaker a display name using
the controls above the tabs; names are annotations stored only in this browser's
local storage, scoped to session ID, and included as `speaker_names` in exports.
They do not establish verified identity or add cross-session voice matching.

Closing the page stops microphone capture and disconnects, but
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

For live snapshots, opt in with `"partials": true` on `start`. Send audio,
then `{"type":"preview","utterance_id":"client-sentence-id"}` without clearing
the current audio. A `partial` response includes that ID, `provisional: true`,
`speaker_status: "pending"`, text and per-model outputs. Busy/throttled previews
receive `partial_skipped`; inference failures receive `partial_error` without
disabling final transcription. Commit with the same client ID to finalize once.
`{"type":"discard"}` clears an unfinished segment and invalidates its draft.

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

### New-speaker confirmation

A sentence whose voice matches no session speaker does not create a speaker
immediately. Its voiceprint is held as *pending* and the sentence is returned
with `speaker.id: null` and `speaker.pending: true` (the tester shows **Speaker
pending confirmation**). When a later unmatched sentence agrees with a pending
voice, Parascribe enrolls a new speaker from both, labels the new sentence, and
relabels the earlier one. The earlier sentence is listed in
`speaker.relabeled` and the stored transcript is updated. A voice heard only
once stays unlabelled. At most `VOICEPRINT_PENDING_LIMIT` pending voices are
kept per session; set `VOICEPRINT_CONFIRM_NEW_SPEAKERS=0` to enroll on the
first sentence instead.

Every match at or above `VOICEPRINT_UPDATE_THRESHOLD` refines the speaker's
centroid, weighted by speech duration, so speakers become easier to recognize as
the session continues.

### Calibration

The defaults come from an offline replay of 883 single-speaker utterances
(0.5–20 s, ground-truth AMI `only_words` RTTMs) from twelve four-speaker AMI
meetings, embedded with the production WeSpeaker model. Within a meeting,
different speakers almost never exceed cosine 0.35 (99th percentile 0.34–0.37),
while same-speaker scores depend strongly on duration (median 0.39 at
0.5–1.5 s, 0.56 at 1.5–3 s, 0.72 at 3–6 s). Replaying each meeting in order:

| Policy | Speakers per 4-speaker meeting | Correctly labelled | Unlabelled |
|---|---:|---:|---:|
| Former defaults (match 0.50, update 0.70) | 21.2 | 64% | 0% |
| Match/update 0.30, enroll immediately | 6.2 | 89% | 0% |
| Match/update 0.30, confirm new speakers | 3.8 | 87% | 8% |

“Correctly labelled” maps each true speaker to their largest predicted speaker.
Close-talk or noisy microphones can shift these scores.

These thresholds are starting points, not universal identity
guarantees. Calibrate them with representative microphones, rooms, languages,
and speakers before using the result for anything consequential. Do not use
voiceprints as authentication.

## Storage and deletion

Session transcripts and voiceprints are stored in `/data/sessions.sqlite3`, a
volume shared by the replicas. Uploaded v2 audio is deleted after processing;
the transcript, anonymous speaker id, job result, and timings remain.

Voiceprints are biometric data. Speaker centroids and pending (unconfirmed)
voices are held in separate tables so the client can delete the vectors while
retaining the transcript; deletion removes both:

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
    "pending": false,
    "relabeled": [],
    "signature_stored": true
  }
}
```

When a sentence's voice is not yet confirmed, `id` is `null` and `pending` is
`true`. When a second sentence confirms a new speaker, `enrolled` is `true`,
`similarity` is its score against the pending voice, and `relabeled` lists the
earlier sentence that now carries the same id. With
`VOICEPRINT_CONFIRM_NEW_SPEAKERS=0`, the first utterance enrolls immediately and
`similarity` is `null`.
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
   If the primary engine recognizes no speech, it sends `no_speech` instead
   (with both engine outputs in `result.transcription`); no transcript entry or
   voiceprint is stored and the stream stays open. REST sentence jobs finish as
   `done` with `result.no_speech: true`.
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
| `VOICEPRINT_SIMILARITY_THRESHOLD` | `0.30` | Minimum cosine similarity for a match or new-speaker confirmation |
| `VOICEPRINT_UPDATE_THRESHOLD` | `0.30` | Minimum similarity before updating a speaker centroid |
| `VOICEPRINT_CONFIRM_NEW_SPEAKERS` | `1` | Require a second agreeing sentence before enrolling a new speaker |
| `VOICEPRINT_PENDING_LIMIT` | `20` | Unconfirmed voices kept per session |
| `STREAM_MAX_UTTERANCE_SECONDS` | `60` | WebSocket commit size limit |
| `SENTENCE_MAX_UPLOAD_BYTES` | `33554432` | REST sentence upload limit |

The current implementation does not resolve overlapping speakers inside one
sentence (a long chunk containing two voices yields a blended voiceprint),
merge accidentally split identities, authenticate clients, or encrypt
the database at rest. Those are the next production-hardening items.
