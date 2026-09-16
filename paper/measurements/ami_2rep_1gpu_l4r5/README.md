# AMI ES2005a — 2 replicas, 1 GPU, concurrency 1–4, 5 repetitions

Measurements backing §\ref{sec:oversub}, §\ref{sec:stages} and
§\ref{sec:capacity} of the paper. Exact configuration is in `meta.json`.

This is an **over-subscription** experiment: the replica count is held at two
and only the offered concurrency varies. It is therefore *not* comparable with
`repetitions/process_short_*.csv`, which varies the replica count with
concurrency matched to it. Level `c=3` here means three jobs against two
replicas, not three replicas.

## Configuration

| | |
|---|---|
| Audio | `testing_audio/dataset/ami/ES2005a.wav` — 477.9 s, 4 speakers, AMI, CC BY 4.0 |
| Cluster | 2 replicas + nginx gateway, `WORKERS=1` each, both on GPU 1 |
| Model | `large-v3`, `float16`, diarization on |
| Levels | offered concurrency 1, 2, 3, 4 — 5 repetitions each, warmed up first |
| Host | 12 vCPU (on an AMD EPYC 9454), 165.1 GiB RAM |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, 97887 MiB, driver 580.173.02 |

`meta.json` records the tree as dirty: the working tree carried uncommitted
`paper/main.tex` edits when the sweep ran. The measurement code itself —
`benchmark_transcribe.py` — was unmodified since `62d8dd0`, so the numbers are
that revision's.

## Files

| File | Rows | What |
|---|---|---|
| `aggregate.csv` | 20 | one row per (concurrency, repetition), 409 columns |
| `jobs.csv` | 50 | one row per job with server stage timings |
| `replicas.csv` | 60 | per-replica CPU/RSS per level |
| `host_samples.csv` | — | raw host series: GPU mem/util/power, cores busy, RAM |
| `container_samples.csv` | — | raw per-container series |
| `meta.json` | — | provenance: commit, driver, image id, model, idle baseline |
| `run.log` | — | console output, including the summary tables |

Every measured quantity in `aggregate.csv` carries `_n _mean _median _sd _cv
_sem _ci95_lo _ci95_hi _p25 _p75 _iqr _p90 _p95 _min _max`, plus
`speedup_vs_c1`, `parallel_efficiency_pct` and `karp_flatt_serial_fraction`.

## Results

| $c$ | files/min | 95% CI | cv | efficiency | Karp–Flatt $e$ |
|---|---|---|---|---|---|
| 1 | 5.896 (0.005) | [5.889, 5.903] | 0.09% | 99.9% | — |
| 2 | 8.382 (0.008) | [8.372, 8.392] | 0.10% | 71.1% | 0.406 |
| 3 | 7.462 (0.313) | [7.073, 7.851] | 4.20% | 45.3% | 0.603 |
| 4 | 8.360 (0.045) | [8.304, 8.416] | 0.54% | 35.6% | 0.604 |

Stage means in seconds (share of `pipeline_s`):

| stage | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| `queue_wait_s` | 0.00 (0%) | 0.00 (0%) | 4.44 (38%) | 6.61 (50%) |
| `transcribe_s` | 1.88 (22%) | 3.47 (26%) | 2.91 (25%) | 3.46 (26%) |
| `align_s` | 3.02 (35%) | 2.99 (22%) | 3.04 (26%) | 2.99 (22%) |
| `diarize_s` | 3.20 (37%) | 6.26 (47%) | 5.26 (44%) | 6.25 (47%) |
| `pipeline_s` | 8.72 | 13.34 | 11.84 | 13.30 |

`prepare_*_s` is 0.00 throughout: caches are mounted and the run is warmed, so
model preparation is excluded by construction.

## Reading the numbers

**Throughput is non-monotonic, and c=4 recovers.** 8.382 at c=2, 7.462 at c=3,
8.360 at c=4 — the c=2 and c=4 intervals overlap. Contention alone would
degrade monotonically. With `R` replicas, `c` jobs run in `ceil(c/R)` waves and
a partial wave costs a full wave's elapsed time, so throughput peaks when
`R | c`. Measured `T_1 = 10.18 s` and `T_2 = 14.31 s` predict `T(3) = 24.49 s`
and `T(4) = 28.63 s` against observed 24.15 s and 28.71 s — −1.4% and +0.3%,
with no fitted parameter.

**c=3 is bimodal.** Wall times are 22.4 s once and 24.5–24.7 s four times, not
scatter about a mean; cv is 4.2% against ≤0.5% elsewhere. The wave model
assumes a wave drains before the next starts, but the trailing job can begin as
soon as the *first* replica frees, which is consistent with the faster cluster.
Observed, not characterised. Do not quote the c=3 mean unqualified.

**Contention is stage-specific.** From c=1 to c=2, transcription grows
1.88 → 3.47 s and diarisation 3.20 → 6.26 s, both near a factor of two, while
alignment is flat at 3.02 → 2.99 s. Two of the three substantial stages
serialise on GPU compute and the third overlaps cleanly — the mechanism behind
the 71% efficiency at c=2. Diarisation at 47% of pipeline time is the largest
single component.

**Karp–Flatt: use the trend, not the magnitude.** At c≥3 the metric charges a
partial wave's idle time to serialisation, which is arithmetically right but
mixes quantisation with true serialisation. The rise from 0.406 to ~0.60
distinguishes growing overhead from a fixed serial section, which speedup alone
cannot; the absolute value is not a property of the pipeline.

**Provisioning: CPU binds before the GPU.** Per concurrent job: 2.39 cores,
3760 MiB GPU, 0.09 GiB host RAM — about 5 concurrent jobs before 12 vCPU are
exhausted, against about 26 before GPU memory is. GPU memory here is
*incremental*; the idle baseline already holds 18 625 MiB of resident weights.
Throughput saturates earlier still, at c=2. Power: 213 W mean, ≈3 Wh per hour
of audio.

## Reproducing

```bash
python3 fetch_models.py
python3 whisperx_cluster.py up --replicas 2 --gpus 1
python3 benchmark_transcribe.py --levels 1,2,3,4 --repeat 5 --warmup \
    --gpu-index 1 --url http://localhost:8357 --out aggregate.csv
```

Sidecars and `meta.json` are written alongside `--out` automatically.

## Relationship to `../repetitions/`

Those files predate the stage instrumentation and carry 16 columns: no stage
breakdown, no dispersion beyond min/mean/max, no CPU or power, no provenance.
They are not comparable with this directory and should not be pooled with it.
