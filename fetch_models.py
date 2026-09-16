#!/usr/bin/env python3
"""
Pre-download every model the WhisperX service needs, skipping whatever is
already cached. Safe to re-run; nothing is re-fetched twice.

Why this exists: only the *libraries* are baked into whisperx-api:local — the
weights are pulled on first use. Without a warm cache the first transcription of
a fresh machine stalls for minutes inside the request, and a gated pyannote repo
fails outright if HF_TOKEN was never accepted. Running this after a build (or
after pruning caches) moves that cost out of the request path.

Why it downloads inside the container: the host does not need torch,
torchaudio, or huggingface_hub, and the versions that resolve the weights are
exactly the ones the service will load. No host Python deps beyond stdlib.

What it fetches:
  faster-whisper <model>    ASR weights            (Systran/faster-whisper-*)
  pyannote segmentation     diarization stage 1    (gated: needs HF_TOKEN)
  pyannote diarization 3.1  diarization pipeline   (gated: needs HF_TOKEN)
  wespeaker voxceleb        speaker embeddings     (pulled by the pipeline)
  torchaudio wav2vec2       forced alignment       (per --language)

Caches (both mounted from the host, so they survive container replacement):
  ~/.cache/huggingface  ->  /workspace/.cache/huggingface   (HF_HOME)
  ~/.cache/torch        ->  /root/.cache/torch              (TORCH_HOME)

Usage:
  python3 fetch_models.py                 # download whatever is missing
  python3 fetch_models.py --check         # report only; exit 1 if anything missing
  python3 fetch_models.py --language es   # also/instead fetch the Spanish aligner
  python3 fetch_models.py --whisper-model large-v2
  python3 fetch_models.py --no-diarize    # skip the gated pyannote repos

Env vars (override defaults):
  WHISPER_MODEL         faster-whisper size         (default large-v3)
  DEFAULT_LANGUAGE      alignment language          (default en)
  HUGGINGFACE_HUB_TOKEN gated-repo token; HF_TOKEN also accepted. Read from the
                        environment, else from .env next to this script.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "whisperx-api:local"
BUILD_CONTEXT = "./whisperx"
HF_CACHE = Path.home() / ".cache" / "huggingface"
TORCH_CACHE = Path.home() / ".cache" / "torch"

PYANNOTE_REPOS = [
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-3.1",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]

# Runs inside the container. Prints one "STATUS<TAB>name<TAB>detail" line per
# model so the host side can render results without parsing library chatter.
PAYLOAD = r'''
import os, sys, traceback

whisper_model = os.environ["FM_WHISPER_MODEL"]
language      = os.environ["FM_LANGUAGE"]
repos         = [r for r in os.environ["FM_REPOS"].split(",") if r]
check_only    = os.environ["FM_CHECK"] == "1"
token         = os.environ.get("HF_TOKEN") or None

missing = 0

def report(status, name, detail=""):
    print(f"STATUS\t{status}\t{name}\t{detail}", flush=True)

def hf_repo(repo):
    """Cached-first: a local_files_only hit means nothing to do."""
    global missing
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, local_files_only=True)
        report("CACHED", repo)
        return
    except Exception:
        pass
    if check_only:
        missing += 1
        report("MISSING", repo)
        return
    try:
        snapshot_download(repo, token=token)
        report("FETCHED", repo)
    except Exception as e:
        missing += 1
        detail = f"{type(e).__name__}: {e}".replace("\n", " ")[:160]
        report("ERROR", repo, detail)

# --- ASR ------------------------------------------------------------------
# faster-whisper resolves bare sizes to the Systran mirror; a user-supplied
# "org/name" is taken as-is.
hf_repo(whisper_model if "/" in whisper_model
        else f"Systran/faster-whisper-{whisper_model}")

# --- diarization ----------------------------------------------------------
for repo in repos:
    hf_repo(repo)

# --- alignment ------------------------------------------------------------
# WhisperX picks torchaudio bundles for some languages and HF models for the
# rest; consult its own tables so we fetch exactly what it will load.
try:
    import whisperx.alignment as alignment
    torch_models = getattr(alignment, "DEFAULT_ALIGN_MODELS_TORCH", {})
    hf_models    = getattr(alignment, "DEFAULT_ALIGN_MODELS_HF", {})

    if language in hf_models:
        hf_repo(hf_models[language])
    elif language in torch_models:
        bundle_name = torch_models[language]
        import torchaudio
        bundle = getattr(torchaudio.pipelines, bundle_name)
        # Bundles expose the checkpoint filename privately; when absent, fall
        # back to letting torchaudio decide whether a download is needed.
        fname = getattr(bundle, "_path", None)
        ckpt = os.path.join(
            os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
            "hub", "checkpoints", fname or "")
        name = f"torchaudio/{bundle_name}"
        if fname and os.path.exists(ckpt):
            report("CACHED", name)
        elif check_only:
            missing += 1
            report("MISSING", name)
        else:
            bundle.get_model()
            report("FETCHED", name)
    else:
        report("SKIP", f"align/{language}",
               "no default aligner; WhisperX falls back at runtime")
except Exception as e:
    missing += 1
    report("ERROR", f"align/{language}",
           f"{type(e).__name__}: {e}".replace("\n", " ")[:160])

sys.exit(1 if missing else 0)
'''


def _token() -> str:
    """Token from the environment, else from .env — same keys the cluster uses."""
    for key in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(("HUGGINGFACE_HUB_TOKEN=", "HF_TOKEN=")):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _have_image() -> bool:
    return subprocess.run(["docker", "image", "inspect", IMAGE],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download WhisperX model weights that are not cached yet.")
    ap.add_argument("--check", action="store_true",
                    help="report what is missing without downloading; "
                         "exit 1 if anything is absent")
    ap.add_argument("--whisper-model",
                    default=os.environ.get("WHISPER_MODEL", "large-v3"),
                    help="faster-whisper size or explicit org/name "
                         "(default %(default)s)")
    ap.add_argument("--language",
                    default=os.environ.get("DEFAULT_LANGUAGE", "en"),
                    help="language whose alignment model to fetch "
                         "(default %(default)s)")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip the gated pyannote repos")
    ap.add_argument("--build", action="store_true",
                    help=f"build {IMAGE} first if it is missing")
    args = ap.parse_args()

    if not _have_image():
        if not args.build:
            print(f"error: image {IMAGE} not found. Build it with:\n"
                  f"  docker build -t {IMAGE} {BUILD_CONTEXT}\n"
                  f"or re-run with --build.", file=sys.stderr)
            return 2
        print(f"Building {IMAGE} ...")
        if subprocess.run(["docker", "build", "-t", IMAGE, BUILD_CONTEXT],
                          cwd=HERE).returncode:
            return 2

    repos = [] if args.no_diarize else PYANNOTE_REPOS
    token = _token()
    if repos and not token:
        print("warning: no HUGGINGFACE_HUB_TOKEN/HF_TOKEN found; the pyannote "
              "repos are gated and will fail. Set it in .env, or pass "
              "--no-diarize.", file=sys.stderr)

    for cache in (HF_CACHE, TORCH_CACHE):
        cache.mkdir(parents=True, exist_ok=True)

    # No --gpus: every fetch is a download, so this never contends with running
    # replicas for GPU memory.
    #
    # `-e HF_TOKEN` with no value forwards it from our own environment instead
    # of writing it into argv, which any user on the host could read out of
    # `ps` for as long as the container runs.
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{HF_CACHE}:/workspace/.cache/huggingface",
        "-v", f"{TORCH_CACHE}:/root/.cache/torch",
        "-e", "HF_HOME=/workspace/.cache/huggingface",
        "-e", "TORCH_HOME=/root/.cache/torch",
        "-e", "HF_TOKEN",
        "-e", f"FM_WHISPER_MODEL={args.whisper_model}",
        "-e", f"FM_LANGUAGE={args.language}",
        "-e", f"FM_REPOS={','.join(repos)}",
        "-e", f"FM_CHECK={'1' if args.check else '0'}",
        "--entrypoint", "python", IMAGE, "-c", PAYLOAD,
    ]
    child_env = dict(os.environ, HF_TOKEN=token)

    verb = "Checking" if args.check else "Fetching"
    print(f"{verb} models for whisper={args.whisper_model} "
          f"language={args.language}"
          f"{' (diarization skipped)' if args.no_diarize else ''}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            env=child_env)
    counts = {}
    for line in proc.stdout:
        if not line.startswith("STATUS\t"):
            continue
        _, status, name, detail = (line.rstrip("\n").split("\t") + [""])[:4]
        counts[status] = counts.get(status, 0) + 1
        mark = {"CACHED": "=", "FETCHED": "+", "MISSING": "!",
                "ERROR": "x", "SKIP": "-"}.get(status, "?")
        print(f"  {mark} {status:<8} {name}" + (f"  {detail}" if detail else ""))
    rc = proc.wait()

    summary = ", ".join(f"{n} {s.lower()}" for s, n in sorted(counts.items()))
    print(f"\n{summary or 'nothing reported'}")
    if rc and args.check:
        print("Run without --check to download the missing models.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
