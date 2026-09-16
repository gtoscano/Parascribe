# AMI ES2005a — 2 replicas, 1 GPU, 5 repetitions

Scaling and resource measurements for the WhisperX gateway cluster, captured
with `benchmark_transcribe.py` at commit `62d8dd0`.

Exact configuration is in `meta.json`; the numbers below are a summary so the
directory is readable without loading the CSVs.

## Configuration

| | |
|---|---|
| Audio | `testing_audio/dataset/ami/ES2005a.wav` — 477.9 s, 4 speakers, AMI, CC BY 4.0 |
| Cluster | 2 replicas + nginx gateway, `WORKERS=1` each, both on GPU 1 |
| Model | `large-v3`, `float16`, diarization on |
| Levels | concurrency 1, 2, 3 — 5 repetitions each, warmed up first |
| Host | 12 vCPU (on an AMD EPYC 9454), 165.1 GiB RAM |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, 97887 MiB, driver 580.173.02 |

## Files

| File | Rows | What |
|---|---|---|
| `aggregate.csv` | 15 | one row per (concurrency, repetition), 409 columns |
| `jobs.csv` | 30 | one row per job with server stage timings |
| `replicas.csv` | 45 | per-replica CPU/RSS per level |
| `host_samples.csv` | 594 | raw host time series: GPU mem/util/power, cores busy, RAM |
| `container_samples.csv` | 237 | raw per-container time series |
| `meta.json` | — | provenance: commit, driver, image id, model, idle baseline |
| `run.log` | — | console output, including the summary tables |

Every measured quantity in `aggregate.csv` carries the full set of suffixes:
`_n _mean _median _sd _cv _sem _ci95_lo _ci95_hi _p25 _p75 _iqr _p90 _p95
_min _max`.

## Results

| conc | files/min | 95% CI | cv | speedup | efficiency | Karp–Flatt *e* |
|---|---|---|---|---|---|---|
| 1 | 5.888 | [5.874, 5.902] | 0.19% | 1.000 | 99.7% | — |
| 2 | 8.382 | [8.372, 8.392] | 0.10% | 1.425 | 71.2% | 0.404 |
| 3 | 7.744 | [7.280, 8.208] | 4.82% | 1.247 | 41.6% | 0.703 |

Stage means, as a share of `pipeline_s`:

| stage | c1 | c2 | c3 |
|---|---|---|---|
| `queue_wait_s` | 0.00 (0%) | 0.00 (0%) | 4.43 (37%) |
| `transcribe_s` | 1.87 (22%) | 3.42 (26%) | 2.90 (25%) |
| `align_s` | 2.94 (34%) | 3.00 (23%) | 3.02 (26%) |
| `diarize_s` | 3.21 (37%) | 6.24 (47%) | 5.27 (45%) |
| `pipeline_s` | 8.66 | 13.28 | 11.81 |

Model preparation (`prepare_*_s`) is 0.00 throughout: the caches are mounted
and the run is warmed up, so it is excluded from these figures by construction.

## Reading the numbers

**Concurrency 3 is bimodal — do not quote its mean alone.** The five
observations are 7.33, 7.34 | 8.01, 8.02, 8.02 files/min, two tight clusters
about 2.1 s of wall clock apart, not scatter around 7.744. The mean falls
between the modes and describes a state the system never occupied; the cv of
4.8% (against 0.1% at c2) is the signal. `queue_wait_s` is flat at ~4.4 s in
both modes, so queueing is not what separates them. Unexplained; report the
median or both modes.

**Karp–Flatt rises (0.404 → 0.703).** The metric is the experimentally
determined serial fraction. Holding roughly constant across levels would
indicate a genuinely serial section; rising indicates overhead that grows with
load — contention and queueing. That distinction is why it is reported
alongside speedup, which cannot separate the two.

**CPU is the binding resource, not the GPU.** Per concurrent job: 3760 MiB GPU,
2.11 cores, 0.09 GiB host RAM. On this host that is ~5 concurrent jobs before
CPU is exhausted, against ~26 before GPU memory is. Measured throughput
nonetheless flattens at concurrency 2, earlier than either ceiling, because
jobs contend for GPU compute.

**`gpu_mem_per_job` is incremental, not resident.** The idle baseline already
includes 18625 MiB held by the replicas' loaded models. Total residency and
per-job marginal cost are different quantities; quote the one you mean.

Power: 216 W mean, ~3 Wh per hour of audio processed.

## Reproducing

```bash
python3 fetch_models.py                       # warm the model caches
python3 whisperx_cluster.py up --replicas 2 --gpus 1
python3 benchmark_transcribe.py --levels 1,2,3 --repeat 5 --warmup \
    --gpu-index 1 --url http://localhost:8357 --out aggregate.csv
```

Sidecar CSVs and `meta.json` are written alongside `--out` automatically.

## Relationship to `../repetitions/`

Those files predate the stage instrumentation and carry 16 columns: no stage
breakdown, no dispersion beyond min/mean/max, no CPU or power, no provenance.
They are not directly comparable with this directory and should not be pooled
with it.
