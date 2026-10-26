# Parascribe

**Scalable speaker-attributed speech transcription on your own GPUs.**

Parascribe is a self-hosted service for batch transcription, sentence/realtime
transcription, and speaker recognition within a conversation.

| API | Transcription | Speaker processing |
|---|---|---|
| `/v1` batch | WhisperX / Whisper `large-v3`, with wav2vec2 alignment | pyannote diarization |
| `/v2` sentences and streams | Parakeet v3 INT8, Cohere Transcribe Q8, or both | WeSpeaker session voiceprints |

Process replicas share a single nginx gateway. The experimental v2 supports
per-session voiceprint deletion and retains both model outputs in comparison
mode. See [`README-v2.md`](README-v2.md) for setup and protocol details, and
[`README-batch.md`](README-batch.md) for the v1 batch API. The performance study
below describes the v1 WhisperX batch pipeline.

For an interactive v2 test, open `/v2/test` on the running service. The
[browser tester instructions](README-v2.md#browser-tester) cover uploads,
microphone streaming, model comparisons, and session voiceprint deletion.
For gpuccino access, safe ports, and a complete testing checklist, see
[TO_TEST.md](TO_TEST.md).

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
clients ──HTTP──▶ nginx gateway (:8357) ──round-robin──▶  parascribe-0  ┐
                        │                                  parascribe-1  ├─ one GPU
                        │  GET /v1/jobs/<i>-… ─▶ replica i  parascribe-2  ┘
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

# 2. Pre-download the model weights (optional but recommended).
#    Only the libraries are baked into the image; weights are pulled on first
#    use, so without this the first request stalls for minutes and a gated
#    pyannote repo fails outright if its licence was never accepted.
python3 fetch_models.py           # fetches whatever is missing, skips the rest
python3 fetch_models.py --check   # report only; exit 1 if anything is absent
python3 fetch_cohere.py           # v2 Cohere weights into models/ (cohere/compare modes)

# 3. Build the image
docker build -t parascribe-api:local ./parascribe

# 4a. Bring up the checked-in topology (1 replica + gateway)
docker compose up -d

# 4b. Or generate a topology for this machine's GPUs (recommended).
#     `up` builds the image itself; `up`/`gen` rewrite docker-compose.yml and
#     parascribe-nginx.conf.
REPLICAS=3 python3 parascribe_cluster.py up            # 3 replicas on one GPU
python3 parascribe_cluster.py up --replicas 3 --gpus 0,1,2,3   # 12 replicas, 4 GPUs
python3 parascribe_cluster.py up --replicas 3 --gpus 0,1 --dry-run   # preview + GPU check
```

The gateway listens on **`:8357`** (single endpoint, load-balanced). The
generated `docker-compose.yml` names its project `parascribe-cluster`, so
plain `docker compose up -d` / `docker compose down` / `docker compose ps` in
the repository root manage the same deployment as
`parascribe_cluster.py {up,down,ps,urls}`.

`docker compose up -d` reuses the existing `parascribe-api:local` image (the
compose file has no `build:` section). After pulling or editing code, rebuild
first with `docker build -t parascribe-api:local ./parascribe`, or use
`parascribe_cluster.py up`.

### Running from scratch

A fresh clone contains only code. Everything else the service needs is created
by the quick-start steps above and is not in git:

```bash
git clone git@github.com:gtoscano/Parascribe.git
cd Parascribe
cp .env.example .env && $EDITOR .env   # Hugging Face token (pyannote licenses accepted)
python3 fetch_models.py                # weights into ~/.cache/huggingface and ~/.cache/torch
python3 fetch_cohere.py                # Cohere weights into models/
docker build -t parascribe-api:local ./parascribe
docker compose up -d
curl http://127.0.0.1:8357/health
```

- **Model caches** are the `~/.cache/huggingface` and `~/.cache/torch` of the
  user who runs `docker compose`. A user without them downloads every model
  again; `fetch_models.py` does this up front instead of on the first request.
- **Session data** starts empty in the clone's `parascribe/data/`. Instances
  started from different clones do not share sessions or transcripts.
- **Root-owned files:** containers run as root, so files they create in
  `parascribe/data/` and `~/.cache/torch` may need `sudo` to remove.

**One instance per server.** All users in the `docker` group share one Docker
daemon, and the project name, container names (`parascribe-0`,
`parascribe-gateway`), ports, and image tag are fixed. Before starting, check
that no other clone's deployment is running (`docker compose ls`) and stop it
with `docker compose down`. Building replaces the shared `parascribe-api:local`
image for every clone on the server.

### Naming migration

The shared deployment now uses `parascribe/`, `parascribe_cluster.py`,
`parascribe-nginx.conf`, image `parascribe-api:local`, and compose project
`parascribe-cluster`. Client settings are `PARASCRIBE_URL` / `PARASCRIBE_URLS`;
GPU selection uses `PARASCRIBE_GPUS` / `PARASCRIBE_GPU`.

The source directory rename preserves its runtime data under `parascribe/data/`.
Existing containers keep their original names until explicitly replaced. Stop
the old deployment before launching the renamed one on the same ports; this
source change does not deploy or restart containers. Published paper commands
and measurement metadata retain their historical names for reproducibility.

### Transcribe

```bash
export PARASCRIBE_URL=http://localhost:8357

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
    --audio your_audio.wav --levels 1,2,3,4 --repeat 5 --warmup
```

`--repeat` runs each concurrency level several times. One pass at level N
produces only N jobs, which is too few to say anything about spread; repeating
gives each level a distribution, and the summary reports a t-based 95% CI
rather than a bare mean.

Five files are written next to `--out`:

| File | Contents |
|------|----------|
| *(the `--out` path)* | one row per (concurrency, repetition) |
| `_jobs.csv` | one row per job, with every server stage timing |
| `_replicas.csv` | per-replica CPU and RSS at each level |
| `_samples.csv` | raw host series: GPU mem/util/power, cores busy, RAM |
| `_container_samples.csv` | raw per-container series |
| `_meta.json` | provenance: commit, GPU driver, image id, model, idle baseline |

Every measured quantity carries `_n _mean _median _sd _cv _sem _ci95_lo
_ci95_hi _p25 _p75 _iqr _p90 _p95 _min _max`, plus `speedup_vs_c1`,
`parallel_efficiency_pct` and `karp_flatt_serial_fraction` per level. Karp–Flatt
is the experimentally determined serial fraction: roughly constant across
levels means a genuinely serial section, rising means overhead that grows with
load (queueing, contention). Speedup alone cannot tell those apart.

The console also prints a **capacity-planning** block — GPU memory, CPU cores
and host RAM per concurrent job, per-replica container footprint, watt-hours
per hour of audio, and which resource binds first as replicas are added. Every
figure is a delta over a measured idle baseline, since a shared host's other
workloads would otherwise be folded into the estimate.

Per-replica CPU/RSS comes from `docker stats`; pass `--no-container-stats` if
the service is not containerized or Docker is not reachable.

Committed results live in `paper/measurements/`. See
[`ami_2rep_1gpu_r5/README.md`](paper/measurements/ami_2rep_1gpu_r5/README.md)
for a worked example — including why that run's concurrency-3 mean should not
be quoted on its own.

---

## Configuration

Set in the compose `environment:` (or via `parascribe_cluster.py`):

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | `large-v3-turbo` is faster, slightly less accurate |
| `COMPUTE_TYPE` | `float16` | `int8` to save VRAM |
| `DEFAULT_LANGUAGE` | `en` | empty = auto-detect |
| `BATCH_SIZE` | `16` | lower on out-of-memory |
| `WORKERS` | `1` | keep at 1 — threads don't help (see paper) |
| `REPLICA_ID` | `0` | set per replica by `parascribe_cluster.py` |

`parascribe_cluster.py` flags: `--replicas` (per GPU), `--gpus 0,1,2,3`,
`--dry-run`, `--no-gpu-check`, `--base-port`, `--gateway-port`.

Full HTTP API, build notes, and troubleshooting are in
[`README-batch.md`](README-batch.md).

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
parascribe/server.py           FastAPI service: v1 batch + v2 sentence/session APIs
parascribe/Dockerfile          CUDA 12.8 + cuDNN 9 image (Blackwell-ready)
parascribe_cluster.py          multi-GPU replica launcher + nginx gateway generator
fetch_models.py              pre-download model weights that are not cached yet
transcribe.py                client (single file or batch)
benchmark_transcribe.py      concurrency/scaling benchmark + capacity planning
test_batch.py             endpoint + end-to-end smoke test
docker-compose.yml           replica + gateway stack (generated by the launcher)
paper/                       LaTeX source + compiled PDF of the measurement study
paper/measurements/          committed benchmark results, one directory per run
README-batch.md           detailed API reference, build notes, troubleshooting
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
