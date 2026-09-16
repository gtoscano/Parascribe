#!/usr/bin/env python3
"""
Test/smoke script for the WhisperX API.

Usage
-----
# Endpoint-only checks (health + models), no audio needed:
  python test_whisperx.py

# Full end-to-end test with your own audio (best test — real speech/speakers):
  python test_whisperx.py path/to/meeting.m4a
  python test_whisperx.py path/to/meeting.m4a --num-speakers 3

# No audio file handy? Auto-generate a 2-speaker clip (needs espeak-ng + ffmpeg):
  python test_whisperx.py --make-sample

Server defaults to http://192.168.2.101:8357 (override with --url or WHISPERX_URL).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

DEFAULT_URL = os.environ.get("WHISPERX_URL", "http://192.168.2.101:8357")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def hr(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# ----------------------------------------------------------------------------
# Step 1 & 2: endpoint health
# ----------------------------------------------------------------------------
def check_health(url):
    hr("STEP 1: Health check")
    try:
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        j = r.json()
        print(f"{PASS} /health -> {j}")
        if not j.get("diarization"):
            print(f"  {FAIL} WARNING: diarization=False (HF_TOKEN not set in the "
                  "container) — speaker labels will be 'Speaker ?'")
        return True
    except Exception as e:
        print(f"{FAIL} Cannot reach {url}/health: {e}")
        print("  Is the container up?  docker compose -f docker-compose-whisperx.yml ps")
        return False


def check_models(url):
    hr("STEP 2: Model info")
    try:
        r = requests.get(f"{url}/v1/models", timeout=10)
        r.raise_for_status()
        print(f"{PASS} /v1/models -> {r.json()}")
        return True
    except Exception as e:
        print(f"{FAIL} /v1/models failed: {e}")
        return False


# ----------------------------------------------------------------------------
# Optional: synthesize a tiny 2-speaker clip so the test is self-contained
# ----------------------------------------------------------------------------
def make_sample():
    if not shutil.which("espeak-ng") and not shutil.which("espeak"):
        sys.exit("espeak-ng/espeak not found. Install it (sudo apt install espeak-ng) "
                 "or pass your own audio file.")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it (sudo apt install ffmpeg) "
                 "or pass your own audio file.")
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    tmp = Path(tempfile.mkdtemp(prefix="whisperx_test_"))
    a, b, out = tmp / "a.wav", tmp / "b.wav", tmp / "sample.wav"
    # Two different voices -> two distinguishable speakers.
    subprocess.run([espeak, "-v", "en-us+m3", "-s", "150",
                    "Hello, thanks for joining the meeting today. Let us begin.",
                    "-w", str(a)], check=True)
    subprocess.run([espeak, "-v", "en-us+f3", "-s", "150",
                    "Sure, happy to be here. I will start with the project update.",
                    "-w", str(b)], check=True)
    # Concatenate the two utterances into one clip.
    lst = tmp / "list.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-ar", "16000", "-ac", "1", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"{PASS} Generated sample clip: {out}")
    return str(out)


# ----------------------------------------------------------------------------
# Step 3: full transcription
# ----------------------------------------------------------------------------
def run_transcription(url, audio, num_speakers=None, min_speakers=None,
                      max_speakers=None, language="en", diarize=True,
                      poll=5, timeout=3600):
    hr(f"STEP 3: Transcribe {Path(audio).name}")
    data = {"language": language, "diarize": str(diarize).lower()}
    for k, v in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                 ("max_speakers", max_speakers)):
        if v:
            data[k] = v

    t0 = time.time()
    with open(audio, "rb") as f:
        r = requests.post(f"{url}/v1/transcribe",
                          files={"file": (Path(audio).name, f)}, data=data,
                          timeout=120)
    if r.status_code != 202:
        print(f"{FAIL} submit failed [{r.status_code}]: {r.text}")
        return False
    job_id = r.json()["job_id"]
    print(f"{PASS} submitted, job_id={job_id}")

    last = None
    while True:
        j = requests.get(f"{url}/v1/jobs/{job_id}", timeout=30).json()
        status, stage = j.get("status"), j.get("stage")
        if stage != last:
            print(f"  [{int(time.time()-t0)}s] {status} / {stage}")
            last = stage
        if status == "done":
            break
        if status == "error":
            print(f"{FAIL} job errored:\n{j.get('error')}")
            return False
        if time.time() - t0 > timeout:
            print(f"{FAIL} timed out after {timeout}s")
            return False
        time.sleep(poll)

    # Validate outputs
    ok = True
    txt = requests.get(f"{url}/v1/jobs/{job_id}/transcript.txt", timeout=30).text
    srt = requests.get(f"{url}/v1/jobs/{job_id}/transcript.srt", timeout=30).text
    res = requests.get(f"{url}/v1/jobs/{job_id}/result.json", timeout=30).json()
    timing = requests.get(
        f"{url}/v1/jobs/{job_id}/timings.json", timeout=30).json()

    print(f"\n--- transcript.txt ---\n{txt.strip()}\n----------------------")

    if not txt.strip():
        print(f"{FAIL} transcript is empty"); ok = False
    else:
        print(f"{PASS} non-empty transcript ({len(txt.split())} words)")

    nspk = res.get("num_speakers", 0)
    if diarize:
        if nspk >= 1 and "Speaker 1" in txt:
            print(f"{PASS} diarization produced {nspk} speaker(s), labels present")
        else:
            print(f"{FAIL} expected 'Speaker N' labels; num_speakers={nspk}"); ok = False
    if not res.get("segments"):
        print(f"{FAIL} result.json has no segments"); ok = False
    else:
        print(f"{PASS} result.json has {len(res['segments'])} segments, "
              f"language={res.get('language')}")
    if srt.strip():
        print(f"{PASS} SRT generated")

    required_timings = {
        "upload_save_s", "queue_wait_s", "load_audio_s", "transcribe_s",
        "align_s", "format_outputs_s", "write_outputs_s", "pipeline_s",
        "processing_s", "persist_metadata_s", "total_server_s",
    }
    durations = timing.get("durations_s", {})
    missing = sorted(required_timings - durations.keys())
    if missing:
        print(f"{FAIL} timings.json missing: {', '.join(missing)}")
        ok = False
    elif res.get("timings") != durations or j.get("timings") != durations:
        print(f"{FAIL} timing data differs across status/result/timings endpoints")
        ok = False
    else:
        print(f"{PASS} persisted {len(durations)} timing measurements "
              f"(server total {durations['total_server_s']:.3f}s)")

    print(f"\nTook {time.time()-t0:.1f}s total")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Test the WhisperX API")
    ap.add_argument("audio", nargs="?", help="audio file for a full end-to-end test")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--make-sample", action="store_true",
                    help="generate a 2-speaker clip with espeak-ng+ffmpeg")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-diarize", action="store_true")
    args = ap.parse_args()

    print(f"Target: {args.url}")
    results = [check_health(args.url), check_models(args.url)]
    if not all(results):
        sys.exit("\nEndpoint checks failed — fix connectivity before testing audio.")

    audio = args.audio
    if not audio and args.make_sample:
        audio = make_sample()

    if audio:
        results.append(run_transcription(
            args.url, audio, num_speakers=args.num_speakers,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            language=args.language, diarize=not args.no_diarize))
    else:
        hr("STEP 3: skipped (no audio)")
        print("Pass an audio file or use --make-sample for a full test.")

    hr("RESULT")
    if all(results):
        print(f"{PASS} ALL CHECKS PASSED")
    else:
        print(f"{FAIL} SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
