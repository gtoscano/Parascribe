#!/usr/bin/env python3
"""Download Voikoro's pinned Cohere Q8 GGUF for Parascribe's optional ASR."""
import argparse
from pathlib import Path
import tempfile
import urllib.request

MODEL = "cohere-transcribe-03-2026-Q8_0.gguf"
REVISION = "ab667acedcb5d8d56837ef3ed3568c3ad50dde47"
URL = (
    "https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/"
    f"{REVISION}/{MODEL}"
)


def valid_model(path):
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return source.read(4) == b"GGUF"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = args.directory / MODEL
    if valid_model(target):
        print(f"Cached: {target}")
        return 0
    if args.check:
        print(f"Missing: {target}")
        return 1
    args.directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pinned Cohere Q8 weights to {target}", flush=True)
    with tempfile.NamedTemporaryFile(dir=args.directory, suffix=".partial", delete=False) as temp:
        partial = Path(temp.name)
    try:
        urllib.request.urlretrieve(URL, partial)
        if not valid_model(partial):
            raise RuntimeError("Downloaded file is not a GGUF model")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
