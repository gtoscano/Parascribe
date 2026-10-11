# Testing Parascribe v2 on gpuccino

Branch: `feature/v2-session-voiceprints`. V1 batch transcription remains
available in the same project; these steps test v2.

## 1. Connect to the running service

Connect your computer to Tailscale and open:

http://gpuccino:8357/v2/test

If the hostname does not resolve, try:

http://100.127.117.17:8357/v2/test

The IP above was checked on 2026-10-11. Obtain the current address on the server
with `tailscale ip -4` if necessary. Include `/v2/test` in the URL.
No Tailscale configuration changes are required or intended.

The service was started and its health endpoint and tester page checked on
2026-10-11. If it is still running, skip the startup section below.

## 2. Check or start the server

Run these commands on gpuccino:

```bash
cd /home/gtoscano/Parascribe
git branch --show-current
docker ps --format 'table {{.Names}}\t{{.Ports}}'
curl --fail --show-error http://127.0.0.1:8357/health
```

The test deployment uses gateway port **8357**, one replica on **18360**, and
GPU **1**. Ports 8357 and 18360 were free before startup. Do not use the
launcher's default replica ports 8360/8361 here: other services use them.

If Parascribe is not running, first check for conflicting listeners:

```bash
ss -ltnp 'sport = :8357'
ss -ltnp 'sport = :18360'
```

An existing Parascribe listener is expected when the service is running. If
another service owns either port, choose unused ports and adjust the URLs and
tunnel below; do not stop that other service.

Preserve the existing `.env`. For a fresh checkout only, copy `.env.example`
to `.env` and configure it as described there. V1 gated diarization models
require a Hugging Face token and license acceptance.

Check the local Cohere weights (download only if missing):

```bash
python3 fetch_cohere.py --check
# If missing:
python3 fetch_cohere.py
```

Start with the explicit safe ports:

```bash
python3 parascribe_cluster.py up --replicas 1 --gpus 1 --gateway-port 8357 --base-port 18360
python3 parascribe_cluster.py ps
curl --fail --show-error http://127.0.0.1:8357/health
curl --fail --show-error -o /dev/null http://127.0.0.1:8357/v2/test
```

Startup builds the image and regenerates Compose/nginx files. It does not
change Tailscale. Models load lazily, so a healthy server does not mean that
every model has already warmed up. First requests may download or load weights.
For errors:

```bash
docker compose -p parascribe-cluster logs --tail 100 parascribe-0 parascribe-gateway
```

## 3. Enable microphone testing without changing Tailscale

Remote HTTP works for uploads, but browsers normally block microphone access
on `http://gpuccino:8357`. On **your own computer**, open an SSH tunnel:

```bash
ssh -N -L 18357:127.0.0.1:8357 YOUR_USERNAME@gpuccino
```

Replace `YOUR_USERNAME` with **your own server account**. Every student runs
this command in Terminal on their own Mac (or another SSH-capable computer),
not in a terminal on gpuccino. Use your own password or SSH key; do not use
another person's account. Port 18357 is local to each student's computer, so
students can use that same local port simultaneously.

Leave that terminal open and visit **http://localhost:18357/v2/test** in your
browser. If local port 18357 is occupied, choose another local port in both the
command and browser URL. This requires working SSH access to gpuccino.
Allow microphone permission when prompted.

The existing Tailscale HTTPS route serves another application; leave it alone.
Do not open the tester HTML directly as a local file.

The SSH login authenticates access to the tunnel, but the current tester/API
does not implement per-account session authorization. Separate session IDs
scope voice matching, not access permissions. Use non-sensitive recordings for
shared classroom testing; do not assume another student cannot access a session.

## 4. Test transcription and both engines

1. For the first transcription-only test, turn off speaker recognition and
   create a session. Save the displayed session ID.
2. Select Parakeet and the recording's language. Upload a short, clear audio
   file containing one speaker. Check the returned text and timings.
3. Select Cohere and upload the same recording. Cohere needs an explicit
   supported language, such as English or Spanish.
4. Select Compare both and upload the same recording. Expect two transcripts
   and separate model timings/real-time factors. The session history uses
   Parakeet as the primary transcript in comparison mode.
5. Download the results before reloading or changing sessions. The export
   includes comparisons collected by the current page. Reopening a session
   restores primary transcripts, not past page comparisons.
6. End the session when finished.

Parakeet uses TDT 0.6B v3 INT8/ONNX Runtime; Cohere uses local Transcribe
03-2026 Q8 through transcribe.cpp. V2 does not use Whisper for transcription.
Voice signatures use WeSpeaker independently of the transcription engine.

For a later accuracy benchmark, retain source audio and human reference
transcripts separately: normal v2 processing deletes uploaded audio. Score
English and Spanish separately, including WER, names, numbers and omissions.
Separate cold loading from warm inference. Compare mode runs the models
sequentially; its combined latency is not either model's standalone latency.

## 5. Test microphone sentence submission

1. Use the localhost tunnel URL, create a session, and select model/language.
2. Click Start microphone, allow access, and speak a sentence.
3. Click Send sentence. Verify a result arrives while recording continues.
4. Speak another sentence and send it; verify history grows.
5. Speak a final sentence and click Stop microphone. Verify the remaining
   audio is submitted. Wait for pending results before changing model modes.

Audio is converted to mono 16 kHz PCM16. These steps describe Manual sentences,
with a 55-second maximum chunk. The optional automatic and provisional-live
behaviors are described below; live snapshots are not native token streaming.

## 6. Test session-local voice signatures

### Automatic and live modes

Before testing speakers, also try both new microphone behaviors:

1. Stop the microphone and wait for pending sentences to finish. Choose
   **Automatic sentences on pause**, then restart. Speak and pause for the
   selected interval (default 0.8 seconds). Confirm sentences appear without
   pressing Send sentence. Silence alone should not create transcript entries.
2. Stop, wait, choose **Live partial text + automatic sentences**, and restart.
   Speak continuously for a few seconds. In **Live transcript**, confirm
   provisional text appears and changes, then a pause creates one final result.
   This periodically re-decodes the unfinished sentence, not native word-token
   streaming; a short sentence may finalize before a preview is ready.
3. Switch to **Final & comparison** for complete model outputs and timings.
   Try Parakeet, Cohere and Compare both, stopping and draining between changes.
   Live mode costs more compute; assess latency as well as accuracy.
4. Adjust pause length and speech sensitivity for your voice/room. Energy-based
   detection can mistake noise for speech; try Quiet voice or Noisy room as
   appropriate. Automatic/live modes also finalize at 20 seconds.

### Speaker matching and names

Obtain participants' consent before capturing their voices.

1. Create session A with speaker recognition enabled. Save its ID.
2. Have person A speak clearly for several seconds. Expect a session-local
   speaker ID, such as `speaker_1`.
3. Have person A say different words. Check whether the ID stays the same.
4. Have person B speak alone, then person A again. Check for distinct IDs and
   consistent matching. Inspect similarities; do not assume a match is proof
   of identity. Try representative microphones, rooms, accents and noise.
5. Stop recording, download results, and create session B with recognition
   enabled. The same person is enrolled independently in B; identical labels
   across sessions do not mean signatures were shared.
6. Reopen session A using its saved ID and confirm its transcript is retained.

Final results prominently show the matched speaker ID. Select it in Session
speaker, enter a display name, and click Set name. Confirm both transcript tabs
update and Download results includes `speaker_names`. Names are session-scoped
annotations saved in this browser, not server-side person identities. Reopening
the session in the same browser should recover them. A live draft shows speaker
matching as pending until the final voice match, never the previous speaker as
a guess. Compare the displayed label against the actual speaker when changing
turns; there is no support yet for overlapping speech or speaker changes within
one submitted sentence.

Each submitted utterance must contain one speaker. Matching is session-local,
not global person identification, and must not be used as authentication.

## 7. Test deletion and ending

In an active recognition-enabled session:

1. Click Delete voice signatures. Confirm the transcript remains.
2. Submit another sentence. Transcription should continue without collecting
   new signatures. Deletion disables capture for that session; use a new
   session to enroll voices again.
3. Optionally verify server state from gpuccino:

```bash
SESSION_ID='paste-your-session-id'
curl --fail --show-error "http://127.0.0.1:8357/v2/sessions/$SESSION_ID"
```

Expect `capture_voiceprints: false`, `speaker_count: 0`, and a non-null
`signatures_deleted_at`. Historical transcript speaker labels may remain;
they are not the voice signature vectors.

In another session with captured voices, leave Delete voice signatures when
ending checked and click End session. Verify status is `ended`, signatures
are deleted, transcript remains, and no new utterances are accepted.
End both sessions from the previous section, deleting their signatures.

Closing a browser tab does **not** end a session or delete signatures.
Operators must separately manage deletion from backups and snapshots.

## 8. Optional automated checks

Run unit/API tests without modifying the running service's data:

```bash
docker run --rm --entrypoint python \
  -e PYTHONDONTWRITEBYTECODE=1 -e DATA_DIR=/tmp/parascribe-tests \
  -v /home/gtoscano/Parascribe:/repo:ro -w /repo \
  parascribe-api:local -m unittest discover -s tests -v
```

The optional browser smoke test requires Python Playwright and its Chromium
browser installed in the test environment, both ASR models configured, and a
short English WAV. Prefer a disposable test deployment; it creates sessions
and uses a synthetic microphone, not a real-person voice-matching benchmark:

```bash
python3 tests/browser_smoke.py --url http://localhost:8357 --audio /absolute/path/speech.wav
```

## Troubleshooting and stopping

- Cannot connect: check server health locally first. If that works but your
  computer cannot connect, check its Tailscale connection and try the IP URL.
  A successful server-side IP check does not verify your client's network path.
- Microphone unavailable: use the localhost tunnel and check browser permission.
- Cohere unavailable: check the weights with `fetch_cohere.py --check`, then
  inspect container logs. A first request may be slow while loading models.
- Voice extraction fails: inspect logs and try clear, longer single-speaker
  speech. Test transcription with recognition disabled to isolate the problem.
- No microphone transcript: click Send sentence or Stop microphone and wait
  for processing; simply speaking does not immediately commit a sentence.

Only when you intend to stop Parascribe:

```bash
python3 parascribe_cluster.py down
```

This leaves persisted session data in place and does not change Tailscale or
stop the other applications on gpuccino.
