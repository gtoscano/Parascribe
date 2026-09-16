# Parascribe

**Scalable speaker-attributed speech transcription on your own GPUs.**

Parascribe is a self-hosted service that turns audio into a **speaker-labeled
transcript** (`Speaker 1:`, `Speaker 2:`, …) using
[WhisperX](https://github.com/m-bain/whisperX) — Whisper `large-v3` for
transcription, wav2vec2 for word-level alignment, and
[pyannote](https://github.com/pyannote/pyannote-audio) for diarization — and
**scales it across GPUs** with a process-replica architecture behind a single
nginx gateway.

It ships with the serving stack, a multi-GPU launcher, a client, a benchmark
suite, and a short paper (`paper/`) that measures the whole thing.

---

## Why it exists

A single transcription stream leaves the GPU **~50 % idle**, because alignment
and diarization are CPU- and Python-bound. The intuitive fix — more worker
*threads* — makes throughput **worse**: the CPython GIL serializes the
Python-heavy stages, so threads contend on the GIL and the GPU.

Parascribe instead runs multiple **process replicas** (each its own interpreter
and CUDA context), which parallelize for real — about **1.7× on one GPU** before
compute saturates, and **linearly across GPUs** (jobs never cross a GPU). A
single nginx gateway fronts every replica; because each replica's job registry
is in-memory, the gateway routes status lookups back to the owning replica by a
prefix encoded in the job id. See `paper/main.pdf` for the full measurement
study.

### Key results (1× NVIDIA RTX PRO 6000)

| Configuration | Throughput (8-min file) | GPU util. |
|---|---|---|
| 1 stream | 5.9 files/min | ~48 % |
| 2 worker **threads** | 4.3 files/min (**regression**) | 35 % |
| 3 process **replicas** | **9.9 files/min** | 86 % |

A batch of ten 40-minute recordings: **~6.6 min** on 1 GPU (3 replicas),
extrapolating to **~1.9 min** on 4 GPUs (12 replicas). Model:
`wall ≈ ⌈N / replicas⌉ × per-job-time`, validated to within 0.5 %.

---

## Architecture

```
clients ──HTTP──▶ nginx gateway (:8357) ──round-robin──▶  whisperx-0  ┐
                        │                                  whisperx-1  ├─ one GPU
                        │  GET /v1/jobs/<i>-… ─▶ replica i  whisperx-2  ┘
                        ▼
                 shared /data volume (transcripts)
```

- **Replica** = one container, `WORKERS=1`, its own ASR/align/diarize model set,
  pinned to a GPU. `k` replicas per GPU (sweet spot ≈ 3).
- **Gateway** = nginx; round-robins `POST /v1/transcribe`, prefix-routes
  `GET /v1/jobs/<id>` back to the owning replica.
- **Multi-GPU** = replicas interleaved across GPUs behind the same gateway.

---

## Quickstart

**Prerequisites:** Docker + the NVIDIA Container Runtime, an NVIDIA GPU, and a
Hugging Face token for diarization.

```bash
# 1. Configure your token (diarization models)
cp .env.example .env
$EDITOR .env          # set HUGGINGFACE_HUB_TOKEN=...
# Accept the pyannote licenses once (links in .env.example).

# 2a. Bring up the checked-in topology as-is (2 replicas + gateway)
docker compose -p whisperx-cluster up -d --build

# 2b. Or generate a topology for this machine's GPUs (recommended).
#     Note: `up`/`gen` rewrite docker-compose.yml and whisperx-nginx.conf.
REPLICAS=3 python3 whisperx_cluster.py up            # 3 replicas on one GPU
python3 whisperx_cluster.py up --replicas 3 --gpus 0,1,2,3   # 12 replicas, 4 GPUs
python3 whisperx_cluster.py up --replicas 3 --gpus 0,1 --dry-run   # preview + GPU check
```

The gateway listens on **`:8357`** (single endpoint, load-balanced). Manage the
cluster with `whisperx_cluster.py {ps,down,urls}`.

### Transcribe

```bash
export WHISPERX_URL=http://localhost:8357

# One file (auto speaker count); saves <file>.transcript.txt next to the audio
python3 transcribe.py meeting.m4a

# Exact speaker count improves accuracy
python3 transcribe.py meeting.m4a --num-speakers 3

# Many files, fanned across replicas via the gateway
python3 transcribe.py *.m4a --num-speakers 8 --concurrency 3
```

Every processed file also gets a `<file>.timings.json` sidecar. It records
client-observed upload, polling, download, local writes, optional summarization,
and total time, plus the server's queue and WhisperX stage durations. Server
copies are retained at `/data/out/<job_id>/timings.json`.

### Benchmark

```bash
python3 benchmark_transcribe.py --url http://localhost:8357 \
    --audio your_audio.wav --levels 1,3,10 --num-speakers 4 --warmup
```

The benchmark writes an aggregate CSV and a companion `_jobs.csv` containing
every individual job's raw client and server-stage timings.

---

## Configuration

Set in the compose `environment:` (or via `whisperx_cluster.py`):

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `large-v3-turbo` is faster, slightly less accurate |
| `COMPUTE_TYPE` | `float16` | `int8` to save VRAM |
| `DEFAULT_LANGUAGE` | `en` | empty = auto-detect |
| `BATCH_SIZE` | `16` | lower on out-of-memory |
| `WORKERS` | `1` | keep at 1 — threads don't help (see paper) |
| `REPLICA_ID` | `0` | set per replica by `whisperx_cluster.py` |

`whisperx_cluster.py` flags: `--replicas` (per GPU), `--gpus 0,1,2,3`,
`--dry-run`, `--no-gpu-check`, `--base-port`, `--gateway-port`.

Full HTTP API, build notes, and troubleshooting are in
[`README-whisperx.md`](README-whisperx.md).

---

## Benchmark data

The paper's numbers use samples from two **public** corpora, which are **not
redistributed here** (only `testing_audio/manifest.csv`, which lists ids,
durations, speaker counts, and licenses, is included):

- **AMI Meeting Corpus** — <https://groups.inf.ed.ac.uk/ami/corpus/>
- **Oyez** (U.S. Supreme Court oral arguments, CC-BY-NC) — <https://www.oyez.org/>

To rebuild the benchmark set from those public sources, run
`testing_audio/build_dataset.py` (needs `requests` + `ffmpeg`, and normal
internet access; the AMI converter's `xmltodict` dependency is installed into
an ignored local directory when needed):

```bash
python3 testing_audio/build_dataset.py all      # download, trim, and write manifest.csv
```

Or bring your own audio: `transcribe.py` and `benchmark_transcribe.py` accept
any file via their arguments (`--audio` / positional). Pass `--audio-dur` to the
benchmark if you are not using the manifest.

---

## Repository layout

```
whisperx/server.py           FastAPI service: transcribe → align → diarize → relabel
whisperx/Dockerfile          CUDA 12.8 + cuDNN 9 image (Blackwell-ready)
whisperx_cluster.py          multi-GPU replica launcher + nginx gateway generator
transcribe.py                client (single file or batch)
benchmark_transcribe.py      concurrency/scaling benchmark
test_whisperx.py             endpoint + end-to-end smoke test
docker-compose.yml           replica + gateway stack (generated by the launcher)
paper/                       LaTeX source + compiled PDF of the measurement study
README-whisperx.md           detailed API reference, build notes, troubleshooting
```

---

## Paper

The measurement study is in [`paper/`](paper/) (`main.pdf`, LaTeX source, and a
reproducibility appendix with the exact benchmark commands). Build with `make`
in that directory (needs TeX Live).

> G. Toscano and G. Connors, *Throughput Scaling of a GPU Speech Transcription
> and Diarization Service: Threads Fail, Processes Win, and a Gateway Ties Them
> Together*, 2026.

---

## License

[MIT](LICENSE) © 2026 Gregorio Toscano and Grace Connors.

Built on WhisperX, faster-whisper / CTranslate2, wav2vec2, pyannote, and
PyTorch — each under its own license.
