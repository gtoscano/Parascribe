#!/usr/bin/env python3
"""
Load/scaling benchmark for the WhisperX transcription server.

Goal
----
Figure out how much hardware you need to transcribe many files at once. We take
ONE audio file, submit it C times "concurrently" for C = 1..N, and measure, for
each concurrency level:

  * wall time            – submit first job  ->  last job done
  * per-job latency      – submit -> done, min / mean / max across the C jobs
  * server process time  – running -> finished (from the server's own timestamps)
  * server stage time    – queue, model prep, ASR, align, diarize, and output stages
  * GPU memory           – peak MiB on the WhisperX GPU, and delta over idle
  * GPU utilization       – mean / peak %, plus estimated GPU-busy seconds
  * host RAM             – peak used GiB during the level
  * throughput           – files/min and real-time-factor (audio-min processed / min)

IMPORTANT — how this server behaves
-----------------------------------
whisperx/server.py runs a SINGLE background worker and holds a global lock
(`_models_lock`) around all GPU work, so jobs are processed ONE AT A TIME. Sending
10 files at once just fills the queue; they still run sequentially on the GPU.

Consequences you'll see in the numbers:
  * GPU memory stays ~flat as concurrency rises (only one job on the GPU at a time).
  * Wall time grows ~linearly with concurrency (queue drains one job at a time).
  * Throughput (files/min) stays ~flat — one server = one job's worth of throughput.

So this benchmark measures the throughput of ONE server instance. To actually run
K files simultaneously you need ~K workers (multiple server instances / GPUs, or a
server refactored to run several jobs in parallel). The summary prints a sizing
estimate based on the measured single-job throughput.

Usage
-----
  # Default: sweep concurrency 1..10 using the shortest AMI file, monitor GPU 1
  python3 benchmark_transcribe.py

  # Pick the audio, speaker count, and levels explicitly
  python3 benchmark_transcribe.py --audio testing_audio/dataset/ami/ES2005a.wav \
      --num-speakers 4 --levels 1,2,4,8,10

  # Quick smoke test (just concurrency 1)
  python3 benchmark_transcribe.py --levels 1

Output:
  The aggregate CSV passed with --out (or benchmark_<timestamp>.csv), plus a
  companion <name>_jobs.csv containing every successful job's raw timings.

Env / flags:
  WHISPERX_URL   default http://localhost:8357   (override with --url)
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

WHISPERX_URL = os.environ.get("WHISPERX_URL", "http://localhost:8357")
DEFAULT_AUDIO = "testing_audio/dataset/ami/ES2005a.wav"  # ~478s, 4 speakers, smallest AMI
SERVER_TIMING_KEYS = [
    "upload_save_s", "queue_wait_s", "load_audio_s", "prepare_asr_model_s",
    "transcribe_s", "prepare_align_model_s", "align_s",
    "prepare_diarize_model_s", "diarize_s", "assign_speakers_s",
    "format_outputs_s", "write_outputs_s", "pipeline_s", "processing_s",
    "persist_metadata_s", "total_server_s",
]


# ---------------------------------------------------------------------------
# Background resource sampler: GPU (nvidia-smi) + host RAM (/proc/meminfo)
# ---------------------------------------------------------------------------
class ResourceSampler(threading.Thread):
    """Polls GPU memory/util for one physical GPU and host RAM at a fixed rate.

    Samples are (timestamp, gpu_mem_MiB, gpu_util_pct, host_ram_used_GiB). Use
    window() to reduce the samples captured between two wall-clock times.
    """

    def __init__(self, gpu_index: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []          # list of tuples
        self._stop = threading.Event()

    def _read_gpu(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits",
                 "-i", str(self.gpu_index)],
                text=True, timeout=5).strip()
            mem, util = (x.strip() for x in out.split(","))
            return float(mem), float(util)
        except Exception:
            return float("nan"), float("nan")

    @staticmethod
    def _read_host_ram_used_gib():
        try:
            total = avail = None
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])          # kB
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])          # kB
                if total is not None and avail is not None:
                    break
            if total is None or avail is None:
                return float("nan")
            return (total - avail) / (1024 * 1024)        # kB -> GiB
        except Exception:
            return float("nan")

    def run(self):
        while not self._stop.is_set():
            mem, util = self._read_gpu()
            ram = self._read_host_ram_used_gib()
            self.samples.append((time.time(), mem, util, ram))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def window(self, t0: float, t1: float):
        return [s for s in self.samples if t0 <= s[0] <= t1]


def _agg(window, idx):
    vals = [s[idx] for s in window if s[idx] == s[idx]]  # drop NaN
    return vals


# ---------------------------------------------------------------------------
# One transcription job: upload + poll to completion
# ---------------------------------------------------------------------------
def submit_and_wait(audio_path: Path, num_speakers, language, diarize,
                    base_url=None, poll=2.0, timeout=7200):
    """Returns a dict of timings for one job (or raises on failure).

    The whole job (submit + poll) stays on one `base_url`, because the job
    registry is per-replica: you must poll the replica you submitted to."""
    base_url = base_url or WHISPERX_URL
    t_submit = time.time()
    data = {"language": language, "diarize": str(diarize).lower()}
    if num_speakers:
        data["num_speakers"] = num_speakers
    with audio_path.open("rb") as f:
        r = requests.post(f"{base_url}/v1/transcribe",
                          files={"file": (audio_path.name, f)}, data=data)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    while True:
        j = requests.get(f"{base_url}/v1/jobs/{job_id}").json()
        status = j.get("status")
        if status == "done":
            t_done = time.time()
            break
        if status == "error":
            raise RuntimeError(f"job {job_id} failed: {j.get('error', '?')[:400]}")
        if time.time() - t_submit > timeout:
            raise TimeoutError(f"job {job_id} timed out")
        time.sleep(poll)

    # server-side processing time from ISO timestamps, if present
    proc = float("nan")
    try:
        s = datetime.fromisoformat(j["started_at"])
        e = datetime.fromisoformat(j["finished_at"])
        proc = (e - s).total_seconds()
    except Exception:
        pass

    return {
        "job_id": job_id,
        "url": base_url,                 # which replica handled it
        "latency": t_done - t_submit,    # client-observed submit->done
        "server_proc": proc,             # server running->finished
        "server_timings": dict(j.get("timings", {})),
        "num_speakers": j.get("num_speakers"),
        "t_submit": t_submit,
        "t_done": t_done,
    }


def run_level(concurrency, audio_path, audio_dur, args, sampler, idle_mem, urls):
    """Run `concurrency` copies of the job at once, round-robined across the
    replica `urls`; return a metrics row."""
    spread = ", ".join(f"{urls[i % len(urls)].rsplit(':', 1)[-1]}"
                       for i in range(concurrency))
    print(f"\n=== concurrency {concurrency} — submitting {concurrency} job(s) "
          f"across {len(urls)} replica(s) [ports {spread}] ===")
    t0 = time.time()
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(submit_and_wait, audio_path, args.num_speakers,
                          args.language, not args.no_diarize,
                          urls[i % len(urls)])
                for i in range(concurrency)]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(str(e))
                print(f"  ! job failed: {e}")
    t1 = time.time()
    wall = t1 - t0

    win = sampler.window(t0, t1)
    mem = _agg(win, 1)
    util = _agg(win, 2)
    ram = _agg(win, 3)

    peak_mem = max(mem) if mem else float("nan")
    mean_util = statistics.mean(util) if util else float("nan")
    peak_util = max(util) if util else float("nan")
    peak_ram = max(ram) if ram else float("nan")
    gpu_busy_s = (mean_util / 100.0) * wall if util else float("nan")

    lat = [r["latency"] for r in results]
    proc = [r["server_proc"] for r in results if r["server_proc"] == r["server_proc"]]
    n_ok = len(results)

    files_per_min = (n_ok / wall * 60) if wall > 0 else float("nan")
    # real-time factor: seconds of audio processed per wall-second
    rtf = (n_ok * audio_dur / wall) if (wall > 0 and audio_dur) else float("nan")

    row = {
        "concurrency": concurrency,
        "jobs_ok": n_ok,
        "jobs_failed": len(errors),
        "wall_s": round(wall, 3),
        "lat_min_s": round(min(lat), 3) if lat else "",
        "lat_mean_s": round(statistics.mean(lat), 3) if lat else "",
        "lat_max_s": round(max(lat), 3) if lat else "",
        "server_proc_mean_s": round(statistics.mean(proc), 3) if proc else "",
    }
    for key in SERVER_TIMING_KEYS:
        values = [r["server_timings"][key] for r in results
                  if key in r["server_timings"]]
        row[f"server_{key.removesuffix('_s')}_mean_s"] = (
            round(statistics.mean(values), 6) if values else "")
    row.update({
        "gpu_peak_mem_MiB": round(peak_mem) if peak_mem == peak_mem else "",
        "gpu_mem_over_idle_MiB": round(peak_mem - idle_mem) if peak_mem == peak_mem else "",
        "gpu_mean_util_pct": round(mean_util, 1) if mean_util == mean_util else "",
        "gpu_peak_util_pct": round(peak_util) if peak_util == peak_util else "",
        "gpu_busy_s": round(gpu_busy_s, 1) if gpu_busy_s == gpu_busy_s else "",
        "host_ram_peak_GiB": round(peak_ram, 2) if peak_ram == peak_ram else "",
        "files_per_min": round(files_per_min, 2) if files_per_min == files_per_min else "",
        "realtime_factor": round(rtf, 2) if rtf == rtf else "",
    })

    job_rows = []
    for result in results:
        job_row = {
            "concurrency": concurrency,
            "job_id": result["job_id"],
            "url": result["url"],
            "client_latency_s": round(result["latency"], 6),
            "server_proc_s": (round(result["server_proc"], 6)
                              if result["server_proc"] == result["server_proc"]
                              else ""),
            "num_speakers": result["num_speakers"],
        }
        for key in SERVER_TIMING_KEYS:
            job_row[key] = result["server_timings"].get(key, "")
        job_rows.append(job_row)

    print(f"  wall={row['wall_s']}s  peak_gpu_mem={row['gpu_peak_mem_MiB']}MiB "
          f"(+{row['gpu_mem_over_idle_MiB']} over idle)  "
          f"mean_util={row['gpu_mean_util_pct']}%  "
          f"host_ram_peak={row['host_ram_peak_GiB']}GiB  "
          f"throughput={row['files_per_min']} files/min  RTF={row['realtime_factor']}x")
    return row, job_rows


def main():
    global WHISPERX_URL
    ap = argparse.ArgumentParser(
        description="Concurrency/scaling benchmark for the WhisperX server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--audio", default=DEFAULT_AUDIO,
                    help="audio file to submit repeatedly")
    ap.add_argument("--audio-dur", type=float, default=None,
                    help="audio duration in seconds (for RTF); read from manifest if omitted")
    ap.add_argument("--levels", default="1,2,3,4,5,6,7,8,9,10",
                    help="comma-separated concurrency levels to test")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="pass --num-speakers to the server (optional)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip speaker diarization")
    ap.add_argument("--gpu-index", type=int, default=1,
                    help="physical GPU index the WhisperX container uses")
    ap.add_argument("--sample-interval", type=float, default=0.5,
                    help="seconds between resource samples")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="seconds to wait between levels")
    ap.add_argument("--warmup", action="store_true",
                    help="run one throwaway job first (loads models / warms caches)")
    ap.add_argument("--url", default=WHISPERX_URL, help="WhisperX server URL")
    ap.add_argument("--urls", default=os.environ.get("WHISPERX_URLS"),
                    help="comma-separated replica URLs to round-robin across "
                         "(env WHISPERX_URLS). Overrides --url. Use with "
                         "whisperx_cluster.py to measure real parallelism.")
    ap.add_argument("--out", default=None,
                    help="CSV output path (default benchmark_<ts>.csv)")
    args = ap.parse_args()

    # Replica pool: --urls (or WHISPERX_URLS) wins, else the single --url.
    if args.urls:
        urls = [u.strip().rstrip("/") for u in args.urls.split(",") if u.strip()]
    else:
        urls = [args.url.rstrip("/")]
    WHISPERX_URL = urls[0]  # default target for helpers that take no base_url

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    # audio duration: CLI > manifest lookup > 0 (RTF disabled)
    audio_dur = args.audio_dur or _duration_from_manifest(audio_path) or 0.0

    try:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"Bad --levels: {args.levels!r}")

    # sanity check every replica
    for u in urls:
        try:
            h = requests.get(f"{u}/health", timeout=10).json()
            print(f"Server OK [{u}]: {h}")
        except Exception as e:
            sys.exit(f"Cannot reach WhisperX at {u}: {e}")

    print(f"\nAudio      : {audio_path}  ({audio_dur or '?'}s, "
          f"{audio_path.stat().st_size/1e6:.1f} MB)")
    print(f"Levels     : {levels}")
    print(f"Replicas   : {len(urls)}  ({', '.join(urls)})")
    print(f"Diarize    : {not args.no_diarize}"
          + (f"  num_speakers={args.num_speakers}" if args.num_speakers else ""))
    print(f"Monitoring : GPU {args.gpu_index} + host RAM, every {args.sample_interval}s")
    print("NOTE: jobs round-robin across replicas; each replica runs one job at "
          "a time (WORKERS=1). Real parallelism needs multiple replica PROCESSES "
          "(GIL-bound pipeline) — watch whether files/min rises with concurrency.\n")

    sampler = ResourceSampler(args.gpu_index, args.sample_interval)
    sampler.start()
    time.sleep(2.0)  # collect a little idle baseline
    idle = _agg(sampler.samples, 1)
    idle_mem = min(idle) if idle else 0.0
    print(f"Idle GPU {args.gpu_index} memory baseline: {idle_mem:.0f} MiB")

    if args.warmup:
        # Warm every replica so each loads its models before we measure.
        print(f"Warmup: 1 job per replica ({len(urls)}) ...")
        with ThreadPoolExecutor(max_workers=len(urls)) as ex:
            wfuts = {ex.submit(submit_and_wait, audio_path, args.num_speakers,
                               args.language, not args.no_diarize, u): u
                     for u in urls}
            for fut in as_completed(wfuts):
                u = wfuts[fut]
                try:
                    fut.result()
                    print(f"  warmup done [{u}]")
                except Exception as e:
                    print(f"  warmup failed [{u}] (continuing): {e}")

    rows, job_rows = [], []
    try:
        for i, c in enumerate(levels):
            level_row, level_jobs = run_level(
                c, audio_path, audio_dur, args, sampler, idle_mem, urls)
            rows.append(level_row)
            job_rows.extend(level_jobs)
            if i < len(levels) - 1 and args.cooldown:
                time.sleep(args.cooldown)
    except KeyboardInterrupt:
        print("\nInterrupted — writing what we have so far.")
    finally:
        sampler.stop()

    if not rows:
        sys.exit("No results collected.")

    out_path = Path(args.out) if args.out else Path(
        f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    jobs_out_path = out_path.with_name(
        f"{out_path.stem}_jobs{out_path.suffix or '.csv'}")
    if job_rows:
        with jobs_out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(job_rows[0].keys()))
            w.writeheader()
            w.writerows(job_rows)

    _print_summary(rows, audio_dur, idle_mem)
    print(f"\nAggregate CSV written: {out_path}")
    if job_rows:
        print(f"Per-job CSV written  : {jobs_out_path}")


def _duration_from_manifest(audio_path: Path):
    """Best-effort lookup of duration_sec from testing_audio/manifest.csv."""
    manifest = Path("testing_audio/manifest.csv")
    if not manifest.exists():
        return None
    stem = audio_path.stem
    try:
        with manifest.open() as f:
            for row in csv.DictReader(f):
                if row.get("id") == stem or Path(row.get("audio_path", "")).stem == stem:
                    return float(row["duration_sec"])
    except Exception:
        pass
    return None


def _print_summary(rows, audio_dur, idle_mem):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    cols = ["concurrency", "wall_s", "lat_mean_s", "lat_max_s", "server_proc_mean_s",
            "gpu_peak_mem_MiB", "gpu_mem_over_idle_MiB", "gpu_mean_util_pct",
            "host_ram_peak_GiB", "files_per_min", "realtime_factor"]
    hdr = ["conc", "wall", "lat_mean", "lat_max", "proc_mean",
           "gpu_peak", "gpu_+idle", "util%", "ram_GiB", "files/min", "RTF"]
    widths = [max(len(h), 9) for h in hdr]
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(w) for c, w in zip(cols, widths)))

    # sizing implication, from the single-job (concurrency=1) throughput
    base = next((r for r in rows if r["concurrency"] == 1), rows[0])
    tp = base.get("files_per_min")
    print("\nInterpretation")
    print("-" * 100)
    if isinstance(tp, (int, float)) and tp:
        print(f"* One server instance sustains ~{tp:.2f} files/min for this audio "
              f"({base.get('realtime_factor','?')}x real-time).")
        print(f"* If files/min stays flat (or drops) as concurrency rises, the "
              f"pipeline is GIL/CPU-bound: threaded workers serialize on the GIL "
              f"and contend for the GPU. Real parallelism needs separate PROCESSES "
              f"(process pool or container replicas), not more threads.")
        print(f"* Peak GPU memory per job ≈ {base.get('gpu_peak_mem_MiB','?')} MiB "
              f"(+{base.get('gpu_mem_over_idle_MiB','?')} MiB over idle). "
              f"A 97 GB GPU could host several worker processes if memory is the only limit "
              f"— but they'd contend for GPU compute.")
    print("* Compare files/min across levels: rising = real parallelism; flat/falling "
          "= bottleneck is CPU/GIL, add processes not threads.")


if __name__ == "__main__":
    main()
