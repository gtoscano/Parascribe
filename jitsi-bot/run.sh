#!/usr/bin/env bash
# Run the transcriber bot in Docker next to a local Parascribe.
#   ./jitsi-bot/run.sh <room or meeting link> [bot options...]
# Transcripts land in ./transcripts (owned by you, not root).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p transcripts
exec docker run --rm -it --init --network host --shm-size 1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e PARASCRIBE_URL="${PARASCRIBE_URL:-http://localhost:8357}" \
  -v "$PWD/transcripts:/transcripts" \
  parascribe-jitsi-bot:local --out /transcripts "$@"
