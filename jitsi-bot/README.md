# Parascribe Jitsi bot

A transcriber that joins a Jitsi Meet room as a visible participant and
transcribes **each participant's microphone separately** through a Parascribe
v2 session. Because Jitsi delivers every person's audio as its own track,
overlapping speech never mixes and each sentence is labelled with the
speaker's Jitsi display name, with no voice signatures needed.

```text
Jitsi room ──► bot: headless Chromium in the meeting page
                 │ one remote audio track per participant
                 ├─ Ana's track  ─► 16 kHz PCM ─► WebSocket ─┐
                 ├─ Luis's track ─► 16 kHz PCM ─► WebSocket ─┼─► Parascribe v2 session
                 └─ ...                                      ─┘   (Parakeet / Cohere)
                                                     transcript.md + .json, live
```

## Quick start (gpuccino)

```bash
# Once, from the repository root:
docker build -f jitsi-bot/Dockerfile -t parascribe-jitsi-bot:local .

# Parascribe must be running (see README-v2.md). Then:
./jitsi-bot/run.sh Synod-Group-3 --language es
./jitsi-bot/run.sh https://meet.jit.si/Synod-Group-3      # a full link also works
```

The transcript is written live to `./transcripts/<room>-<UTC time>.md` (plus a
`.json` with timestamps and participant ids), and each sentence also prints to
the terminal. Stop with Ctrl-C. The bot also leaves when the meeting ends or
the room has been empty for `--empty-timeout` seconds (default 300). On exit
it commits any sentence in progress, waits for the results, and ends the
Parascribe session.

### meet.jit.si needs a host

meet.jit.si only starts a meeting once a **signed-in host** joins. Start the
bot whenever you like: it shows *waiting for the meeting host* and joins
automatically once the host signs in (up to `--join-timeout`, default 30 min).
It cannot start a meeting by itself. A self-hosted Jitsi server does not have
this restriction: pass `--server https://your.jitsi` or a full link.

## Consent and privacy

- The bot appears in the participant list as **Parascribe (transcriber)** and
  posts a bilingual (English/Spanish) notice in the meeting chat when it joins.
  Change the notice with `--announce "..."`, or skip it with `--announce ""`.
- Speaker labels are the names people type into Jitsi. They are user-chosen,
  not verified identities. For a pseudonymous transcript, ask participants to
  join with pseudonyms.
- Voice signatures are **off** by default (`capture_voiceprints: false`); Jitsi
  already says who is speaking. `--voiceprints` turns them on, and they are
  deleted when the bot ends the session.
- Audio is never written to disk by the bot; Parascribe deletes each sentence's
  audio after transcription. Transcripts (`transcripts/`) are git-ignored.
- The bot cannot hear rooms that use end-to-end encryption.

## Options

| Option | Default | Purpose |
|---|---|---|
| `room` | (required) | Room name, or a full meeting link |
| `--server` | `https://meet.jit.si` | Jitsi server for a bare room name |
| `--parascribe` | `$PARASCRIBE_URL` or `http://localhost:8357` | Parascribe gateway |
| `--language` | auto (Parakeet) | `es`, `en`, …; required for `cohere`/`compare` |
| `--asr-mode` | `parakeet` | `parakeet`, `cohere`, or `compare` |
| `--name` | `Parascribe (transcriber)` | Bot display name |
| `--announce` | bilingual notice | Chat message on joining (`""` to skip) |
| `--voiceprints` | off | Also capture session voice signatures |
| `--out` | `transcripts` | Transcript directory |
| `--threshold` / `--pause` / `--max-seconds` | 0.015 / 0.8 / 20 | Sentence detection, as in the browser tester |
| `--join-timeout` / `--empty-timeout` | 1800 / 300 | Seconds |
| `--keep-session-open` | off | Leave the Parascribe session active on exit |
| `--headed`, `-v` | off | Debugging |

## How it works

1. Playwright opens the meeting with URL overrides: no pre-join screen,
   microphone and camera muted, no incoming video (`channelLastN=0`), and no
   peer-to-peer mode, so all audio comes through the Jitsi bridge.
2. Once joined, `capture.js` reads each remote audio track with
   `MediaStreamTrackProcessor`. It converts the audio to mono 16 kHz PCM16 with
   the same averaging resampler as `parascribe/web/pcm-worklet.js`, and passes
   it to Python every 200 ms. (AudioWorklet does not load in headless Chromium
   in Docker, so the bot does not use Web Audio.)
3. `endpointing.py` (a port of the browser tester's `SentenceGate`) splits
   sentences on pauses. A sentence also ends when its speaker goes silent
   (muted tracks deliver no audio) or when a track is handed to another person.
4. `streams.py` keeps one Parascribe WebSocket per track. Each `commit` carries
   `speaker_label` (the Jitsi display name) and `spoken_at` (when the sentence
   began), which Parascribe stores with the utterance.
5. `transcript.py` rewrites the transcript in spoken order after every
   sentence, so even an interrupted bot leaves a complete file.

On meet.jit.si the bridge forwards only the most active speakers' audio and
reassigns a received track to whoever is speaking (SSRC rewriting). The bot
reads the track's current owner for every frame, so sentences stay attributed
correctly. Several people talking at the same moment are captured as long as
the bridge forwards them, normally the three or so loudest.

`capture.js` uses the Jitsi Meet web app's `APP.conference._room` object. It is
stable in practice but internal, so a Jitsi Meet update could require changes.

## Tests

```bash
# Unit, protocol (fake Parascribe) and browser-capture tests (simulated room):
docker run --rm --shm-size 1g --entrypoint python -w /app/tests \
  parascribe-jitsi-bot:local -m unittest test_units test_streams test_capture

# End to end against a running Parascribe: two Oyez speakers on separate
# tracks with an overlapping turn, real capture and transcription (Jitsi simulated):
docker run --rm --network host --shm-size 1g \
  -v "$PWD/testing_audio/dataset/oyez:/audio:ro" --entrypoint python -w /app/tests \
  parascribe-jitsi-bot:local fake_meeting_e2e.py \
  --parascribe http://localhost:8357 --wav /audio/2009.08-1107-t01.wav
```

The end-to-end script creates one Parascribe session (no voiceprints) and ends it.
