# Parascribe project memory

Recorded 2026-10-10.

- User prefers efficient transcription models consistent with Voikoro and their other projects. Do not default new realtime work to Whisper.
- Parascribe v2 uses Parakeet TDT 0.6B v3, INT8 through ONNX Runtime on CPU, for sentence and realtime transcription.
- Voice signatures use `pyannote/wespeaker-voxceleb-resnet34-LM` (WeSpeaker ResNet34-LM). Match speakers only within their conversation/session; names require a client-provided association.
- Store voice signatures per session, with no cross-session identity matching. Clients can request their deletion at session end while preserving transcripts.
- The existing v1 batch API uses WhisperX with its configured Whisper model.
- User chose to keep both Parakeet and Cohere and compare their accuracy later with a benchmark (2026-10-11).
- V2 now implements `parakeet`, `cohere`, and `compare` request modes for REST and WebSockets. Cohere uses local `transcribe-cpp` and Voikoro's pinned Q8 GGUF weights. Cohere modes require an explicit language and a configured model file. Compare stores both outputs/timings and keeps Parakeet as the session transcript.
- V2 work was committed as `3f5aea3` on `feature/v2-session-voiceprints`. The implementation uses the same project with versioned APIs.
- User requested generic service naming to avoid confusing deployment infrastructure with the WhisperX model (2026-10-11). Source directory is `parascribe/`, launcher `parascribe_cluster.py`, nginx config `parascribe-nginx.conf`, image `parascribe-api:local`, compose project `parascribe-cluster`, and client environment variables `PARASCRIBE_URL` / `PARASCRIBE_URLS`. V1 reference is `README-batch.md` and smoke client is `test_batch.py`. Actual WhisperX library references remain for v1. Historical paper artifacts keep their original names.
- Browser microphone modes now include manual, automatic sentences on pause (energy endpointing), and live provisional snapshot re-decoding plus automatic finalization. This is not native token streaming. Drafts are ephemeral, do not write transcripts or enroll voices, and become stale on commit/discard. Automatic chunks cap at 20 seconds; manual at 55. Speaker-threshold calibration and neural VAD remain follow-up work.
- User wants the current speaker clearly visible. Final sentences have prominent session speaker badges; live drafts show speaker pending until final matching. Browser-only session-scoped display names are saved in localStorage and exported as `speaker_names`; they are user annotations, not verified identities. Live and Final & comparison are separate in-page tabs.
- A browser tester is served at `/v2/test`, with source under `parascribe/web/`. It supports session creation/reconnection, audio uploads, manual microphone sentence commits, all ASR modes, result export, and voiceprint deletion. Microphone use needs HTTPS or localhost (an SSH tunnel works for remote testing).

- User explicitly requests no Tailscale configuration changes. Access Parascribe at `http://gpuccino:8357/v2/test`; microphone testing can use an SSH tunnel to localhost. On 2026-10-11 the service was started with one replica on GPU 1, gateway 8357 and direct replica 18360, avoiding existing services on 8360/8361. Full steps are in `TO_TEST.md`.
- Students have their own server accounts. SSH instructions must use `YOUR_USERNAME@gpuccino`, not the owner's account, and state that students run the tunnel locally on their own computers.

Read this file when continuing Parascribe model or session work. Verify the current code and branch before treating implementation details as current.
