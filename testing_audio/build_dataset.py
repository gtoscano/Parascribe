#!/usr/bin/env python3
"""
Builds a labeled test set of (audio, transcript) pairs for ASR/transcription
benchmarking, covering two conversational styles:

  - structured  : SCOTUS oral arguments (Oyez) -- one-at-a-time, ordered turns
  - dialogue    : AMI Meeting Corpus -- overlapping, general multi-party talk

Run on a machine with normal internet access (this needs oyez S3 buckets and
groups.inf.ed.ac.uk, which are NOT reachable from sandboxed/offline environments).

Requirements:
    pip install requests xmltodict
    ffmpeg must be installed and on PATH (for trimming audio to 20 min)

Usage:
    python build_dataset.py select-oyez        # pick candidate SCOTUS cases -> oyez_candidates.csv
    python build_dataset.py download-oyez       # download+trim audio, extract matching transcript
    python build_dataset.py setup-ami           # download AMI annotations + clone converter, build JSON transcripts
    python build_dataset.py download-ami        # download+trim AMI audio for selected meetings
    python build_dataset.py manifest            # build final manifest.csv over everything downloaded
    python build_dataset.py all                 # run all of the above in order
"""

import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
LOCAL_PYTHON_DEPS = ROOT / ".python-deps"
CLIP_SECONDS = 40 * 60  # cap clips at 40 minutes; shorter recordings are used in full

# ---------------------------------------------------------------------------
# Oyez (structured, one-at-a-time speech)
# ---------------------------------------------------------------------------

OYEZ_REPO = "https://github.com/walkerdb/supreme_court_transcripts.git"
OYEZ_REPO_DIR = ROOT / "supreme_court_transcripts"

MIN_SPEAKERS = 4
MAX_SPEAKERS = 8
MIN_YEAR = 2000       # modern recordings = cleaner audio
MIN_TOTAL_DURATION_MIN = 20  # so we can safely clip a 20-min window
NUM_OYEZ_CASES = 12


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def ensure_local_python_dependency(import_name, package_name=None):
    """Install a missing package locally without modifying the system Python."""
    if importlib.util.find_spec(import_name) is not None:
        return None

    if ((LOCAL_PYTHON_DEPS / import_name).exists()
            or (LOCAL_PYTHON_DEPS / f"{import_name}.py").exists()):
        return LOCAL_PYTHON_DEPS

    package_name = package_name or import_name
    print(f"installing {package_name} into {LOCAL_PYTHON_DEPS}...")
    run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--target", str(LOCAL_PYTHON_DEPS), package_name,
    ])
    return LOCAL_PYTHON_DEPS


def write_plain_text(turns, txt_path):
    """
    Writes a plain-text transcript in the form:
        Speaker 1: text
        Speaker 2: text
        ...
    Speaker labels are normalized to Speaker 1, Speaker 2, etc., numbered in
    order of first appearance (so real names / AMI letter-codes don't leak
    into the ground-truth text your transcription code is scored against).
    """
    speaker_num = {}
    lines = []
    for t in turns:
        raw = t["speaker"]
        if raw not in speaker_num:
            speaker_num[raw] = len(speaker_num) + 1
        lines.append(f"Speaker {speaker_num[raw]}: {t['text']}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def select_oyez():
    if not OYEZ_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", OYEZ_REPO, str(OYEZ_REPO_DIR)])
    else:
        run(["git", "-C", str(OYEZ_REPO_DIR), "pull"])

    files = list((OYEZ_REPO_DIR / "oyez" / "cases").glob("*-t*.json"))
    print(f"scanning {len(files)} case files (this takes 1-3 min, no network calls)...")
    candidates = []

    for i, f in enumerate(files):
        if i and i % 1000 == 0:
            print(f"  ...{i}/{len(files)} scanned, {len(candidates)} matches so far")
        m = re.search(r"/(\d{4})\.", str(f))
        if not m or int(m.group(1)) < MIN_YEAR:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue

        sections = (d.get("transcript") or {}).get("sections") or []
        if not sections:
            continue

        max_stop = 0.0
        speakers = set()
        for sec in sections:
            for turn in sec.get("turns", []):
                sp = turn.get("speaker")
                if sp and sp.get("name"):
                    speakers.add(sp["name"])
                for tb in turn.get("text_blocks", []):
                    if tb.get("stop"):
                        max_stop = max(max_stop, tb["stop"])

        media = d.get("media_file") or []
        mp3 = next((m["href"] for m in media if m and m.get("mime") == "audio/mpeg"), None)
        if not mp3:
            continue

        dur_min = max_stop / 60.0
        if dur_min >= MIN_TOTAL_DURATION_MIN and MIN_SPEAKERS <= len(speakers) <= MAX_SPEAKERS:
            candidates.append({
                "case_json": str(f.relative_to(ROOT)),
                "title": d.get("title", ""),
                "duration_min": round(dur_min, 1),
                "num_speakers": len(speakers),
                "mp3_url": mp3,
            })

    candidates.sort(key=lambda c: -c["num_speakers"])  # prefer more speakers first
    chosen = candidates[:NUM_OYEZ_CASES]

    out_csv = ROOT / "oyez_candidates.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_json", "title", "duration_min", "num_speakers", "mp3_url"])
        w.writeheader()
        w.writerows(chosen)

    print(f"\nFound {len(candidates)} total candidates, wrote top {len(chosen)} to {out_csv}")
    for c in chosen:
        print(f"  [{c['num_speakers']} spk, {c['duration_min']} min] {c['title']}")


def download_oyez():
    import requests  # local import so `select-oyez` doesn't require it

    csv_path = ROOT / "oyez_candidates.csv"
    if not csv_path.exists():
        sys.exit("Run `select-oyez` first.")

    out_dir = DATA / "oyez"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        case_json_path = ROOT / row["case_json"]
        d = json.load(open(case_json_path))
        case_id = case_json_path.stem  # e.g. 2016.15-1189-t01

        wav_out = out_dir / f"{case_id}.wav"
        mp3_tmp = out_dir / f"{case_id}.mp3"
        txt_out = out_dir / f"{case_id}.transcript.json"
        plain_out = out_dir / f"{case_id}.txt"

        if wav_out.exists():
            print(f"skip audio (exists): {case_id}")
        else:
            print(f"downloading audio: {row['title']}")
            resp = requests.get(row["mp3_url"], stream=True, timeout=60)
            resp.raise_for_status()
            with open(mp3_tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)

            # trim to first CLIP_SECONDS and convert to wav (16kHz mono, common ASR input)
            run([
                "ffmpeg", "-y", "-i", str(mp3_tmp), "-t", str(CLIP_SECONDS),
                "-ac", "1", "-ar", "16000", str(wav_out)
            ])
            mp3_tmp.unlink()

        # transcript generation is cheap/local (no network) -- always (re)build it,
        # so a missing .json/.txt gets backfilled without re-downloading audio
        turns_out = []
        for sec in d["transcript"]["sections"]:
            for turn in sec.get("turns", []):
                sp = turn.get("speaker")
                speaker_name = sp["name"] if sp and sp.get("name") else "UNKNOWN"
                blocks = [tb for tb in turn.get("text_blocks", []) if tb.get("stop", 0) <= CLIP_SECONDS]
                if not blocks:
                    continue
                text = " ".join(b["text"] for b in blocks)
                turns_out.append({
                    "speaker": speaker_name,
                    "start": blocks[0].get("start"),
                    "stop": blocks[-1].get("stop"),
                    "text": text,
                })

        with open(txt_out, "w") as f:
            json.dump({
                "case_id": case_id,
                "title": row["title"],
                "source": "oyez",
                "source_url": f"https://api.oyez.org/cases",
                "license": "CC-BY-NC (Oyez, Inc.)",
                "style": "structured_one_at_a_time",
                "num_speakers": int(row["num_speakers"]),
                "clip_seconds": CLIP_SECONDS,
                "turns": turns_out,
            }, f, indent=2)

        write_plain_text(turns_out, plain_out)

        print(f"  -> {wav_out.name}, {txt_out.name}, {plain_out.name}")


# ---------------------------------------------------------------------------
# AMI (general/overlapping dialogue)
# ---------------------------------------------------------------------------

AMI_ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_AUDIO_BASE = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_CONVERTER_REPO = "https://github.com/guokan-shang/ami-and-icsi-corpora.git"
AMI_CONVERTER_DIR = ROOT / "ami-and-icsi-corpora"

# Scenario meetings with 4 participants (A,B,C,D) -- picked across different
# sessions/rooms for variety. Add/remove IDs freely; see:
# https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml for the full list.
AMI_MEETING_IDS = [
    "ES2002a", "ES2002b", "ES2003a", "ES2004a", "ES2005a",
    "IS1000a", "IS1001a", "IS1002b", "IS1003a", "IS1004a",
    "TS3003a", "TS3004a",
]


def setup_ami():
    import requests

    ROOT.mkdir(exist_ok=True)
    zip_path = ROOT / "ami_public_manual_1.6.2.zip"
    if not zip_path.exists():
        print("downloading AMI manual annotations (~22MB)...")
        resp = requests.get(AMI_ANNOTATIONS_URL, stream=True, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    if not AMI_CONVERTER_DIR.exists():
        run(["git", "clone", "--depth", "1", AMI_CONVERTER_REPO, str(AMI_CONVERTER_DIR)])

    input_dir = AMI_CONVERTER_DIR / "ami-corpus" / "input" / "ami_public_manual_1.6.2"
    if not input_dir.exists():
        input_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["unzip", "-q", str(zip_path), "-d", str(input_dir)])

    local_deps = ensure_local_python_dependency("xmltodict")
    converter_env = os.environ.copy()
    if local_deps is not None:
        existing_pythonpath = converter_env.get("PYTHONPATH")
        converter_env["PYTHONPATH"] = (
            str(local_deps) if not existing_pythonpath
            else str(local_deps) + os.pathsep + existing_pythonpath
        )

    print("converting NXT annotations -> per-meeting JSON transcripts...")
    run(
        [sys.executable, "dialogueActs.py"],
        cwd=str(AMI_CONVERTER_DIR / "ami-corpus"),
        env=converter_env,
    )
    print(f"done. JSON transcripts in {AMI_CONVERTER_DIR / 'ami-corpus' / 'output' / 'dialogueActs'}")


def download_ami():
    import requests

    out_dir = DATA / "ami"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = AMI_CONVERTER_DIR / "ami-corpus" / "output" / "dialogueActs"

    if not transcripts_dir.exists():
        sys.exit("Run `setup-ami` first.")

    for meeting_id in AMI_MEETING_IDS:
        src_json = transcripts_dir / f"{meeting_id}.json"
        if not src_json.exists():
            print(f"skip {meeting_id}: no transcript produced (check the meeting ID exists)")
            continue

        wav_out = out_dir / f"{meeting_id}.wav"
        wav_full = out_dir / f"{meeting_id}.full.wav"
        txt_out = out_dir / f"{meeting_id}.transcript.json"
        plain_out = out_dir / f"{meeting_id}.txt"

        if wav_out.exists():
            print(f"skip audio (exists): {meeting_id}")
        else:
            url = f"{AMI_AUDIO_BASE}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
            print(f"downloading audio: {meeting_id}")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(wav_full, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

            run(["ffmpeg", "-y", "-i", str(wav_full), "-t", str(CLIP_SECONDS),
                 "-ac", "1", "-ar", "16000", str(wav_out)])
            wav_full.unlink()

        # transcript generation is cheap/local (no network) -- always (re)build it,
        # so a missing .json/.txt gets backfilled without re-downloading audio
        dacts = json.load(open(src_json))
        clipped = [d for d in dacts if float(d.get("endtime", 0) or 0) <= CLIP_SECONDS]
        speakers = sorted(set(d["speaker"] for d in clipped))

        with open(txt_out, "w") as f:
            json.dump({
                "meeting_id": meeting_id,
                "source": "AMI",
                "source_url": "https://groups.inf.ed.ac.uk/ami/corpus/",
                "license": "CC BY 4.0",
                "style": "general_dialogue",
                "num_speakers": len(speakers),
                "clip_seconds": CLIP_SECONDS,
                "turns": [
                    {"speaker": d["speaker"], "start": float(d["starttime"]),
                     "stop": float(d["endtime"]), "text": d["text"]}
                    for d in clipped
                ],
            }, f, indent=2)

        plain_turns = [{"speaker": d["speaker"], "text": d["text"]} for d in clipped]
        write_plain_text(plain_turns, plain_out)

        print(f"  -> {wav_out.name}, {txt_out.name}, {plain_out.name}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def get_audio_duration_sec(wav_path):
    """Reads actual duration from the audio file itself via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True, check=True,
    )
    return round(float(result.stdout.strip()), 1)


def build_manifest():
    rows = []
    for style_dir in ["oyez", "ami"]:
        d = DATA / style_dir
        if not d.exists():
            continue
        for txt_path in sorted(d.glob("*.transcript.json")):
            meta = json.load(open(txt_path))
            wav_path = txt_path.with_name(txt_path.name.replace(".transcript.json", ".wav"))
            plain_path = txt_path.with_name(txt_path.name.replace(".transcript.json", ".txt"))
            if not wav_path.exists():
                continue
            try:
                duration_sec = get_audio_duration_sec(wav_path)
            except Exception as e:
                print(f"warning: couldn't read duration for {wav_path.name}: {e}")
                duration_sec = ""
            rows.append({
                "id": wav_path.stem,
                "source": meta["source"],
                "style": meta["style"],
                "num_speakers": meta["num_speakers"],
                "duration_sec": duration_sec,
                "audio_path": str(wav_path.relative_to(ROOT)),
                "transcript_json_path": str(txt_path.relative_to(ROOT)),
                "transcript_txt_path": str(plain_path.relative_to(ROOT)) if plain_path.exists() else "",
                "license": meta["license"],
            })

    out_csv = ROOT / "manifest.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "style", "num_speakers", "duration_sec",
                                           "audio_path", "transcript_json_path",
                                           "transcript_txt_path", "license"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_csv} with {len(rows)} audio+transcript pairs")
    by_style = {}
    for r in rows:
        by_style[r["style"]] = by_style.get(r["style"], 0) + 1
    for style, n in by_style.items():
        print(f"  {style}: {n}")


if __name__ == "__main__":
    steps = {
        "select-oyez": select_oyez,
        "download-oyez": download_oyez,
        "setup-ami": setup_ami,
        "download-ami": download_ami,
        "manifest": build_manifest,
    }
    if len(sys.argv) != 2 or (sys.argv[1] not in steps and sys.argv[1] != "all"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "all":
        for name, fn in steps.items():
            print(f"\n=== {name} ===")
            fn()
    else:
        steps[sys.argv[1]]()
