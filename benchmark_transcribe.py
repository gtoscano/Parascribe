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
import json
import math
import os
import platform
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
# Suffixes emitted for every measured quantity. Means alone hide the thing a
# scaling benchmark is actually about — whether a level is stable or merely
# averaging out a bimodal mix of fast and queued jobs.
STAT_SUFFIXES = ["n", "mean", "median", "sd", "cv", "sem", "ci95_lo", "ci95_hi",
                 "p25", "p75", "iqr", "p90", "p95", "min", "max"]

# Two-sided 95% t critical values by degrees of freedom. A benchmark level
# rarely has more than a handful of repetitions, where the normal approximation
# understates the interval badly (t=4.30 vs 1.96 at n=3).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t95(df):
    return _T95.get(df, 1.96) if df >= 1 else float("nan")


def _percentile(values, q):
    """Linear-interpolation percentile; q in [0,100]. Defined for n == 1."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _stats(values, nd=3):
    """Summary statistics for one measured quantity.

    `cv` (sd/mean) is the headline number for run-to-run stability: it is
    scale-free, so a 0.02 on a 9s stage and a 0.02 on a 0.1s stage mean the
    same thing, and it makes an unstable level obvious without eyeballing sd
    against magnitude.
    """
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return {s: "" for s in STAT_SUFFIXES}
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    half = _t95(n - 1) * sem if n > 1 else float("nan")
    p25, p75 = _percentile(vals, 25), _percentile(vals, 75)
    return {
        "n": n,
        "mean": round(mean, nd),
        "median": round(statistics.median(vals), nd),
        "sd": round(sd, nd),
        "cv": round(sd / mean, 4) if mean else "",
        "sem": round(sem, nd),
        # Blank rather than a bogus point interval when n == 1: a single
        # observation has no interval, and writing mean±0 would invite a
        # reviewer to read precision that was never measured.
        "ci95_lo": round(mean - half, nd) if half == half else "",
        "ci95_hi": round(mean + half, nd) if half == half else "",
        "p25": round(p25, nd),
        "p75": round(p75, nd),
        "iqr": round(p75 - p25, nd),
        "p90": round(_percentile(vals, 90), nd),
        "p95": round(_percentile(vals, 95), nd),
        "min": round(min(vals), nd),
        "max": round(max(vals), nd),
    }


def _flat_stats(prefix, values, nd=3):
    """`_stats` flattened into CSV columns: prefix_mean, prefix_sd, ..."""
    return {f"{prefix}_{k}": v for k, v in _stats(values, nd).items()}


def _add_scaling_columns(rows):
    """Speedup, parallel efficiency and the Karp-Flatt serial fraction.

    Karp-Flatt: e = (1/S - 1/p) / (1 - 1/p), with S the measured speedup at p
    concurrent jobs. It is the *experimentally determined* serial fraction, and
    it is the right statistic here because it separates the two reasons
    throughput flattens: a roughly constant e across p means a genuinely serial
    section (each job's GPU work serializing), while an e that climbs with p
    means growing overhead — queueing, contention, scheduling. Reporting
    speedup alone cannot distinguish them.
    """
    baseline = [r for r in rows
                if r.get("concurrency") == 1
                and isinstance(r.get("files_per_min"), (int, float))]
    if not baseline:
        return
    base_fpm = statistics.mean(r["files_per_min"] for r in baseline)
    if not base_fpm:
        return
    for r in rows:
        fpm, p = r.get("files_per_min"), r.get("concurrency")
        if not isinstance(fpm, (int, float)) or not p:
            continue
        speedup = fpm / base_fpm
        r["speedup_vs_c1"] = round(speedup, 3)
        r["parallel_efficiency_pct"] = round(100 * speedup / p, 1)
        if p > 1 and speedup > 0:
            e = (1 / speedup - 1 / p) / (1 - 1 / p)
            r["karp_flatt_serial_fraction"] = round(e, 4)
        else:
            r["karp_flatt_serial_fraction"] = ""


# ---------------------------------------------------------------------------
# Background resource sampler: GPU (nvidia-smi) + host RAM (/proc/meminfo)
# ---------------------------------------------------------------------------
class ResourceSampler(threading.Thread):
    """Polls one physical GPU plus host CPU/RAM at a fixed rate.

    Samples are (timestamp, gpu_mem_MiB, gpu_util_pct, host_ram_used_GiB,
    cpu_cores_busy, gpu_power_W). CPU is reported in *cores busy* rather than
    a percentage: a percentage means nothing without knowing the core count,
    whereas cores map directly onto what a deployment has to provision.

    Use window() to reduce the samples captured between two wall-clock times.
    """

    def __init__(self, gpu_index: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []          # list of tuples
        self._stop = threading.Event()
        self._cpu_prev = None      # (busy_jiffies, total_jiffies)
        self.ncpu = os.cpu_count() or 1

    def _read_gpu(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits",
                 "-i", str(self.gpu_index)],
                text=True, timeout=5).strip()
            mem, util, power = (x.strip() for x in out.split(","))
            # power.draw reads "[N/A]" on cards that do not report it.
            try:
                power_w = float(power)
            except ValueError:
                power_w = float("nan")
            return float(mem), float(util), power_w
        except Exception:
            return float("nan"), float("nan"), float("nan")

    @staticmethod
    def gpu_capacity(gpu_index):
        """(name, total_MiB) for the GPU under test — the denominator for
        'how many replicas fit'."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(gpu_index)],
                text=True, timeout=5).strip()
            name, total = (x.strip() for x in out.split(","))
            return name, float(total)
        except Exception:
            return "?", float("nan")

    def _read_cpu_cores_busy(self):
        """Cores busy since the previous sample, from /proc/stat deltas."""
        try:
            fields = Path("/proc/stat").read_text().split("\n")[0].split()[1:]
            vals = [int(x) for x in fields]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            prev = self._cpu_prev
            self._cpu_prev = (total - idle, total)
            if prev is None:
                return float("nan")
            busy_d = (total - idle) - prev[0]
            total_d = total - prev[1]
            if total_d <= 0:
                return float("nan")
            return (busy_d / total_d) * self.ncpu
        except Exception:
            return float("nan")

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
        self._read_cpu_cores_busy()   # prime the delta; first read is unusable
        while not self._stop.is_set():
            mem, util, power = self._read_gpu()
            ram = self._read_host_ram_used_gib()
            cores = self._read_cpu_cores_busy()
            self.samples.append((time.time(), mem, util, ram, cores, power))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def window(self, t0: float, t1: float):
        return [s for s in self.samples if t0 <= s[0] <= t1]


class ContainerSampler(threading.Thread):
    """Per-replica CPU and RSS via `docker stats`.

    Host-wide numbers cannot be divided by replica count when other workloads
    share the box, and this machine runs several unrelated stacks. Sampling the
    containers by name gives a footprint that is actually attributable.

    `docker stats --no-stream` costs ~1s per call, so this runs on its own
    slower clock rather than blocking the main sampler.
    """

    def __init__(self, name_filter="whisperx", interval: float = 2.0):
        super().__init__(daemon=True)
        self.name_filter = name_filter
        self.interval = interval
        self.samples = []          # (timestamp, name, cpu_cores, mem_GiB)
        self._stop = threading.Event()

    def _read(self):
        try:
            out = subprocess.check_output(
                ["docker", "stats", "--no-stream", "--format",
                 "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                text=True, timeout=20)
        except Exception:
            return []
        now, rows = time.time(), []
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or self.name_filter not in parts[0]:
                continue
            name, cpu_s, mem_s = parts
            try:
                # docker reports CPU as a percentage of one core, so 250% is
                # 2.5 cores; convert so it composes with the host figure.
                cores = float(cpu_s.strip().rstrip("%")) / 100.0
            except ValueError:
                cores = float("nan")
            rows.append((now, name, cores, _parse_docker_size(
                mem_s.split("/")[0].strip())))
        return rows

    def run(self):
        while not self._stop.is_set():
            self.samples.extend(self._read())
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def window(self, t0: float, t1: float):
        return [s for s in self.samples if t0 <= s[0] <= t1]


def _parse_docker_size(text):
    """'1.234GiB' / '567MiB' / '12.3kB' -> GiB."""
    text = text.strip()
    units = {"B": 1 / 1024**3, "KB": 1 / 1024**2, "KIB": 1 / 1024**2,
             "MB": 1 / 1024, "MIB": 1 / 1024, "GB": 1.0, "GIB": 1.0,
             "TB": 1024.0, "TIB": 1024.0}
    for suffix in sorted(units, key=len, reverse=True):
        if text.upper().endswith(suffix):
            try:
                return float(text[:-len(suffix)]) * units[suffix]
            except ValueError:
                return float("nan")
    return float("nan")


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


def run_level(concurrency, audio_path, audio_dur, args, sampler, idle_mem, urls,
              repetition=1, idle_cores=float("nan"), idle_ram=float("nan"),
              csampler=None):
    """Run `concurrency` copies of the job at once, round-robined across the
    replica `urls`; return a metrics row."""
    spread = ", ".join(f"{urls[i % len(urls)].rsplit(':', 1)[-1]}"
                       for i in range(concurrency))
    rep_label = f" rep {repetition}" if repetition > 1 or args.repeat > 1 else ""
    print(f"\n=== concurrency {concurrency}{rep_label} — submitting "
          f"{concurrency} job(s) across {len(urls)} replica(s) "
          f"[ports {spread}] ===")
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
    cores = _agg(win, 4)
    power = _agg(win, 5)

    peak_mem = max(mem) if mem else float("nan")
    mean_util = statistics.mean(util) if util else float("nan")
    peak_util = max(util) if util else float("nan")
    peak_ram = max(ram) if ram else float("nan")
    gpu_busy_s = (mean_util / 100.0) * wall if util else float("nan")
    # Energy for this level: mean draw over the wall clock. Useful for cost
    # per hour of audio, which is what deployment budgets are written in.
    energy_wh = (statistics.mean(power) * wall / 3600.0) if power else float("nan")

    lat = [r["latency"] for r in results]
    proc = [r["server_proc"] for r in results if r["server_proc"] == r["server_proc"]]
    n_ok = len(results)

    files_per_min = (n_ok / wall * 60) if wall > 0 else float("nan")
    # real-time factor: seconds of audio processed per wall-second
    rtf = (n_ok * audio_dur / wall) if (wall > 0 and audio_dur) else float("nan")

    row = {
        "concurrency": concurrency,
        "repetition": repetition,
        "jobs_ok": n_ok,
        "jobs_failed": len(errors),
        "wall_s": round(wall, 3),
        # Retained under their original names so existing CSV consumers and the
        # committed paper/measurements files keep working.
        "lat_min_s": round(min(lat), 3) if lat else "",
        "lat_mean_s": round(statistics.mean(lat), 3) if lat else "",
        "lat_max_s": round(max(lat), 3) if lat else "",
        "server_proc_mean_s": round(statistics.mean(proc), 3) if proc else "",
    }
    row.update(_flat_stats("lat", lat))
    row.update(_flat_stats("server_proc", proc))
    for key in SERVER_TIMING_KEYS:
        values = [r["server_timings"][key] for r in results
                  if key in r["server_timings"]]
        stem = f"server_{key.removesuffix('_s')}"
        row[f"{stem}_mean_s"] = (
            round(statistics.mean(values), 6) if values else "")
        row.update(_flat_stats(stem, values, nd=6))
        # Where the pipeline actually spends its time, which is the comparison
        # worth making across levels; absolute seconds move with audio length.
        pipe = [r["server_timings"].get("pipeline_s") for r in results
                if r["server_timings"].get("pipeline_s")]
        row[f"{stem}_pct_of_pipeline"] = (
            round(100 * statistics.mean(values) / statistics.mean(pipe), 2)
            if values and pipe else "")
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
    # --- capacity-planning block -----------------------------------------
    row.update(_flat_stats("cpu_cores", cores, nd=2))
    row.update(_flat_stats("gpu_util_pct", util, nd=1))
    row.update(_flat_stats("gpu_mem_MiB", mem, nd=0))
    row.update(_flat_stats("host_ram_GiB", ram, nd=2))
    row.update(_flat_stats("gpu_power_W", power, nd=1))
    row["gpu_energy_Wh"] = round(energy_wh, 2) if energy_wh == energy_wh else ""
    row["cpu_cores_over_idle"] = (
        round(max(cores) - idle_cores, 2)
        if cores and idle_cores == idle_cores else "")
    row["host_ram_over_idle_GiB"] = (
        round(peak_ram - idle_ram, 2)
        if peak_ram == peak_ram and idle_ram == idle_ram else "")
    # Per-concurrent-job cost: the numbers you multiply when sizing a target
    # throughput. Divided by concurrency, not by replica count, because a
    # level may queue rather than run all jobs at once.
    if concurrency:
        row["gpu_mem_per_job_MiB"] = (
            round((peak_mem - idle_mem) / concurrency)
            if peak_mem == peak_mem else "")
        row["cpu_cores_per_job"] = (
            round((max(cores) - idle_cores) / concurrency, 2)
            if cores and idle_cores == idle_cores else "")
        row["host_ram_per_job_GiB"] = (
            round((peak_ram - idle_ram) / concurrency, 2)
            if peak_ram == peak_ram and idle_ram == idle_ram else "")
    # Per-replica container footprint, attributable rather than inferred.
    replica_rows = []
    if csampler is not None:
        cwin = csampler.window(t0, t1)
        names = sorted({s[1] for s in cwin})
        per_replica_cpu, per_replica_mem = [], []
        for name in names:
            c_vals = [s[2] for s in cwin if s[1] == name and s[2] == s[2]]
            m_vals = [s[3] for s in cwin if s[1] == name and s[3] == s[3]]
            if c_vals:
                per_replica_cpu.append(max(c_vals))
            if m_vals:
                per_replica_mem.append(max(m_vals))
            # One row per process per level: lets an unevenly loaded replica
            # show up instead of being averaged away by the gateway's
            # round-robin, which only balances counts, not work.
            rrow = {
                "concurrency": concurrency,
                "repetition": repetition,
                "container": name,
                "jobs_on_replica": sum(
                    1 for r in results
                    if name.endswith(str(r["job_id"].split("-")[0]))),
                "samples": len(c_vals),
            }
            rrow.update(_flat_stats("cpu_cores", c_vals, nd=2))
            rrow.update(_flat_stats("ram_GiB", m_vals, nd=2))
            replica_rows.append(rrow)
        row["containers_sampled"] = len(names)
        row["container_cpu_cores_peak_sum"] = (
            round(sum(per_replica_cpu), 2) if per_replica_cpu else "")
        row["container_cpu_cores_peak_max"] = (
            round(max(per_replica_cpu), 2) if per_replica_cpu else "")
        row["container_ram_GiB_peak_sum"] = (
            round(sum(per_replica_mem), 2) if per_replica_mem else "")
        row["container_ram_GiB_peak_max"] = (
            round(max(per_replica_mem), 2) if per_replica_mem else "")
        # Spread across replicas: near 0 means the gateway balanced the work,
        # a large value means one process carried the level.
        row["container_cpu_imbalance"] = (
            round(max(per_replica_cpu) - min(per_replica_cpu), 2)
            if len(per_replica_cpu) > 1 else "")

    job_rows = []
    for result in results:
        job_row = {
            "concurrency": concurrency,
            "repetition": repetition,
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
    return row, job_rows, replica_rows


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
    ap.add_argument("--container-interval", type=float, default=2.0,
                    help="seconds between per-container `docker stats` samples")
    ap.add_argument("--container-filter", default="whisperx",
                    help="substring matching the container names to attribute "
                         "CPU/RAM to")
    ap.add_argument("--no-container-stats", action="store_true",
                    help="skip per-container sampling (no docker access, or "
                         "the service is not containerized)")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="seconds to wait between levels")
    ap.add_argument("--repeat", type=int, default=1,
                    help="times to repeat each concurrency level. One pass at "
                         "level N yields N jobs, which is too few to say "
                         "anything about spread; repeating gives per-level "
                         "run-to-run statistics")
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
    csampler = None
    if not args.no_container_stats:
        csampler = ContainerSampler(args.container_filter,
                                    args.container_interval)
        csampler.start()
    # A longer idle window than the old 2s: CPU and RAM baselines are noisier
    # than GPU memory, and everything below is reported as a delta over these.
    time.sleep(max(4.0, args.sample_interval * 6))
    idle = _agg(sampler.samples, 1)
    idle_mem = min(idle) if idle else 0.0
    idle_cores_vals = _agg(sampler.samples, 4)
    idle_cores = statistics.median(idle_cores_vals) if idle_cores_vals else float("nan")
    idle_ram_vals = _agg(sampler.samples, 3)
    idle_ram = min(idle_ram_vals) if idle_ram_vals else float("nan")

    gpu_name, gpu_total = ResourceSampler.gpu_capacity(args.gpu_index)
    total_ram = float("nan")
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_ram = int(line.split()[1]) / (1024 * 1024)
                break
    except Exception:
        pass
    print(f"\nHost capacity : {sampler.ncpu} CPU cores, {total_ram:.1f} GiB RAM")
    print(f"GPU {args.gpu_index}        : {gpu_name}, {gpu_total:.0f} MiB total")
    print(f"Idle baseline : GPU {idle_mem:.0f} MiB, "
          f"{idle_cores:.2f} cores busy, {idle_ram:.2f} GiB RAM"
          + ("  (other workloads share this host — deltas are what matter)"
             if idle_cores == idle_cores and idle_cores > 1 else ""))

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

    rows, job_rows, replica_rows = [], [], []
    passes = [(c, rep) for c in levels for rep in range(1, args.repeat + 1)]
    try:
        for i, (c, rep) in enumerate(passes):
            level_row, level_jobs, level_replicas = run_level(
                c, audio_path, audio_dur, args, sampler, idle_mem, urls,
                repetition=rep, idle_cores=idle_cores, idle_ram=idle_ram,
                csampler=csampler)
            rows.append(level_row)
            job_rows.extend(level_jobs)
            replica_rows.extend(level_replicas)
            if i < len(passes) - 1 and args.cooldown:
                time.sleep(args.cooldown)
    except KeyboardInterrupt:
        print("\nInterrupted — writing what we have so far.")
    finally:
        sampler.stop()
        if csampler is not None:
            csampler.stop()

    if not rows:
        sys.exit("No results collected.")

    _add_scaling_columns(rows)

    out_path = Path(args.out) if args.out else Path(
        f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def _sidecar(suffix, rows_):
        """Write <stem>_<suffix>.csv next to the aggregate; return its path."""
        if not rows_:
            return None
        path = out_path.with_name(
            f"{out_path.stem}_{suffix}{out_path.suffix or '.csv'}")
        # Rows can differ in keys (a replica absent from one level), so union
        # the fieldnames rather than trusting the first row.
        fields = list(dict.fromkeys(k for r in rows_ for k in r))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_)
        return path

    jobs_out_path = _sidecar("jobs", job_rows)
    replicas_out_path = _sidecar("replicas", replica_rows)

    # Raw time series, so resource curves can be replotted without re-running
    # a benchmark that takes minutes and perturbs a shared host.
    sample_rows = [
        {"t": round(s[0], 3), "gpu_mem_MiB": s[1], "gpu_util_pct": s[2],
         "host_ram_GiB": round(s[3], 3) if s[3] == s[3] else "",
         "cpu_cores_busy": round(s[4], 3) if s[4] == s[4] else "",
         "gpu_power_W": s[5] if s[5] == s[5] else ""}
        for s in sampler.samples]
    samples_out_path = _sidecar("samples", sample_rows)

    container_rows = [
        {"t": round(s[0], 3), "container": s[1],
         "cpu_cores": round(s[2], 3) if s[2] == s[2] else "",
         "ram_GiB": round(s[3], 3) if s[3] == s[3] else ""}
        for s in (csampler.samples if csampler else [])]
    container_out_path = _sidecar("container_samples", container_rows)

    meta_path = out_path.with_name(f"{out_path.stem}_meta.json")
    meta = _provenance(
        args, urls, audio_path, audio_dur, gpu_name, gpu_total, sampler.ncpu,
        total_ram,
        {"gpu_mem_MiB": round(idle_mem),
         "cpu_cores_busy": round(idle_cores, 3) if idle_cores == idle_cores else None,
         "host_ram_GiB": round(idle_ram, 3) if idle_ram == idle_ram else None})
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    _print_summary(rows, audio_dur, idle_mem)
    _print_replica_profile(replica_rows)
    _print_capacity(rows, audio_dur, gpu_name, gpu_total, sampler.ncpu,
                    total_ram)
    print(f"\nAggregate CSV written : {out_path}")
    for label, path in (("Per-job", jobs_out_path),
                        ("Per-replica", replicas_out_path),
                        ("Host samples", samples_out_path),
                        ("Container samples", container_out_path)):
        if path:
            print(f"{label+' CSV':<22}: {path}")
    print(f"{'Provenance JSON':<22}: {meta_path}")


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


def _print_across_repetitions(rows):
    """Run-to-run spread per level. Only meaningful with --repeat > 1, where a
    wide cv says the level is not reproducible and its mean should not be
    quoted without one."""
    by_level = {}
    for r in rows:
        by_level.setdefault(r["concurrency"], []).append(r)
    if all(len(v) < 2 for v in by_level.values()):
        return
    print("\nAcross repetitions (throughput stability per level)")
    print("-" * 100)
    hdr = ["conc", "reps", "fpm_mean", "fpm_sd", "fpm_cv", "ci95_lo",
           "ci95_hi", "speedup", "eff%", "karp_e"]
    widths = [max(len(h), 9) for h in hdr]
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for level in sorted(by_level):
        group = by_level[level]
        fpm = [r["files_per_min"] for r in group
               if isinstance(r.get("files_per_min"), (int, float))]
        f = _stats(fpm)
        # Scaling columns are per-row and identical within a level; take the
        # first that carries them.
        sc = next((r for r in group if "speedup_vs_c1" in r), {})
        vals = [level, len(group), f["mean"], f["sd"], f["cv"], f["ci95_lo"],
                f["ci95_hi"], sc.get("speedup_vs_c1", ""),
                sc.get("parallel_efficiency_pct", ""),
                sc.get("karp_flatt_serial_fraction", "")]
        print("  ".join(str(v).rjust(w) for v, w in zip(vals, widths)))
    print("  karp_e = Karp-Flatt serial fraction: flat across levels means a "
          "genuinely serial\n  section; rising means growing overhead "
          "(queueing/contention).")


def _print_stage_profile(rows):
    """Where pipeline time goes per level — the view that shows which stage
    absorbs contention as concurrency rises, rather than only that it did."""
    stages = ["queue_wait_s", "load_audio_s", "transcribe_s", "align_s",
              "diarize_s", "assign_speakers_s", "pipeline_s"]
    levels = sorted({r["concurrency"] for r in rows})
    print("\nServer stage profile — mean seconds (share of pipeline)")
    print("-" * 100)
    print(f"{'stage':<22}" + "".join(f"{'c'+str(c):>17}" for c in levels))
    for stage in stages:
        stem = f"server_{stage.removesuffix('_s')}"
        line = f"{stage:<22}"
        for level in levels:
            group = [r for r in rows if r["concurrency"] == level]
            vals = [r.get(f"{stem}_mean") for r in group
                    if isinstance(r.get(f"{stem}_mean"), (int, float))]
            pcts = [r.get(f"{stem}_pct_of_pipeline") for r in group
                    if isinstance(r.get(f"{stem}_pct_of_pipeline"), (int, float))]
            if not vals:
                line += f"{'-':>17}"
                continue
            mean = statistics.mean(vals)
            share = f"{statistics.mean(pcts):.0f}%" if pcts else "-"
            line += f"{mean:>11.2f} {share:>5}"
        print(line)


def _provenance(args, urls, audio_path, audio_dur, gpu_name, gpu_total,
                ncpu, total_ram, idle):
    """Everything needed to say what produced these numbers.

    A results table is not reproducible without the code revision, the model,
    the driver and the image actually running. Collected automatically because
    it is exactly the metadata that gets reconstructed from memory months later
    when the paper is written.
    """
    def _sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    here = Path(__file__).resolve().parent
    meta = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "cpu_cores": ncpu,
            "cpu_model": next(
                (l.split(":", 1)[1].strip()
                 for l in Path("/proc/cpuinfo").read_text().splitlines()
                 if l.startswith("model name")), None)
            if Path("/proc/cpuinfo").exists() else None,
            "total_ram_GiB": round(total_ram, 2) if total_ram == total_ram else None,
        },
        "gpu": {
            "index": args.gpu_index,
            "name": gpu_name,
            "total_MiB": gpu_total if gpu_total == gpu_total else None,
            "driver": _sh(["nvidia-smi", "--query-gpu=driver_version",
                           "--format=csv,noheader", "-i", str(args.gpu_index)]),
        },
        "code": {
            "git_commit": _sh(["git", "-C", str(here), "rev-parse", "HEAD"]),
            "git_dirty": bool(_sh(["git", "-C", str(here), "status",
                                   "--porcelain"])),
        },
        "service": {
            "urls": urls,
            "image_id": _sh(["docker", "image", "inspect", "whisperx-api:local",
                             "--format", "{{.Id}}"]),
            "image_created": _sh(["docker", "image", "inspect",
                                  "whisperx-api:local", "--format",
                                  "{{.Created}}"]),
        },
        "workload": {
            "audio": str(audio_path),
            "audio_duration_s": audio_dur,
            "audio_bytes": audio_path.stat().st_size if audio_path.exists() else None,
            "language": args.language,
            "diarize": not args.no_diarize,
            "num_speakers": args.num_speakers,
            "levels": args.levels,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
        },
        "idle_baseline": idle,
    }
    # Whatever the servers report about themselves, verbatim.
    try:
        meta["service"]["health"] = [
            requests.get(f"{u}/health", timeout=10).json() for u in urls]
        meta["service"]["models"] = [
            requests.get(f"{u}/v1/models", timeout=10).json() for u in urls]
    except Exception:
        pass
    return meta


def _print_replica_profile(replica_rows):
    """Per-process footprint, so an unevenly loaded replica is visible."""
    if not replica_rows:
        return
    print("\nPer-replica process footprint (peak of each, by level)")
    print("-" * 100)
    hdr = ["conc", "rep", "container", "cpu_mean", "cpu_peak", "ram_mean",
           "ram_peak"]
    widths = [6, 5, 20, 10, 10, 10, 10]
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for r in sorted(replica_rows,
                    key=lambda x: (x["concurrency"], x["repetition"],
                                   x["container"])):
        vals = [r["concurrency"], r["repetition"], r["container"],
                r.get("cpu_cores_mean", ""), r.get("cpu_cores_max", ""),
                r.get("ram_GiB_mean", ""), r.get("ram_GiB_max", "")]
        print("  ".join(str(v).rjust(w) for v, w in zip(vals, widths)))


def _print_capacity(rows, audio_dur, gpu_name, gpu_total, ncpu, total_ram):
    """Turn the measurements into provisioning numbers.

    Everything here is a delta over the idle baseline, because this host runs
    other workloads; absolute peaks would bake their footprint into the
    estimate.
    """
    usable = [r for r in rows
              if isinstance(r.get("files_per_min"), (int, float))]
    if not usable:
        return
    best = max(usable, key=lambda r: r["files_per_min"])

    print("\n" + "=" * 100)
    print("CAPACITY PLANNING")
    print("=" * 100)
    print(f"Measured on: {gpu_name} ({gpu_total:.0f} MiB), "
          f"{ncpu} cores, {total_ram:.1f} GiB RAM")
    print(f"Audio unit : {audio_dur:.0f}s per file\n")

    def val(row, key, default=None):
        v = row.get(key)
        return v if isinstance(v, (int, float)) else default

    gpu_job = val(best, "gpu_mem_per_job_MiB")
    cpu_job = val(best, "cpu_cores_per_job")
    ram_job = val(best, "host_ram_per_job_GiB")
    print(f"Per concurrent job, at the best level (concurrency "
          f"{best['concurrency']}):")
    print(f"  GPU memory     {gpu_job if gpu_job is not None else '?'} MiB")
    print(f"  CPU            {cpu_job if cpu_job is not None else '?'} cores")
    print(f"  Host RAM       {ram_job if ram_job is not None else '?'} GiB")

    c_cpu = val(best, "container_cpu_cores_peak_max")
    c_ram = val(best, "container_ram_GiB_peak_max")
    if c_cpu is not None or c_ram is not None:
        print(f"\nPer replica container (peak of any one replica):")
        print(f"  CPU            {c_cpu if c_cpu is not None else '?'} cores")
        print(f"  RSS            {c_ram if c_ram is not None else '?'} GiB")

    print(f"\nThroughput     {best['files_per_min']:.2f} files/min "
          f"({best.get('realtime_factor','?')}x real-time) at concurrency "
          f"{best['concurrency']}")
    power = val(best, "gpu_power_W_mean")
    if power and audio_dur and best["files_per_min"]:
        audio_h_per_h = best["files_per_min"] * 60 * audio_dur / 3600
        wh_per_audio_h = power / audio_h_per_h if audio_h_per_h else float("nan")
        print(f"Power          {power:.0f} W mean -> ~{wh_per_audio_h:.0f} Wh "
              f"per hour of audio processed")

    # Headroom: which resource runs out first as replicas are added.
    print("\nScaling headroom on this host (binding constraint first):")
    limits = []
    if gpu_job and gpu_job > 0:
        limits.append(("GPU memory", int(gpu_total // gpu_job)))
    if cpu_job and cpu_job > 0:
        limits.append(("CPU cores", int(ncpu // cpu_job)))
    if ram_job and ram_job > 0:
        limits.append(("Host RAM", int(total_ram // ram_job)))
    for name, n in sorted(limits, key=lambda x: x[1]):
        print(f"  {name:<14} supports ~{n} concurrent job(s)")
    if limits:
        binding, n = min(limits, key=lambda x: x[1])
        print(f"\n  -> {binding} binds first, at ~{n} concurrent jobs.")
        print(f"     Note this is a *memory/CPU* ceiling, not a throughput one: "
              f"measured scaling already flattens at concurrency "
              f"{best['concurrency']} because jobs contend for GPU compute. "
              f"Size to the measured throughput, not to this ceiling.")

    print("\nTo serve a target load:")
    fpm = best["files_per_min"]
    if fpm and audio_dur:
        for target_h in (1, 10, 100):
            audio_s = target_h * 3600
            files = audio_s / audio_dur
            minutes = files / fpm
            print(f"  {target_h:>4}h of audio -> {minutes:7.1f} min on this "
                  f"configuration ({minutes/60:.2f} h)")


def _print_summary(rows, audio_dur, idle_mem):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    cols = ["concurrency", "repetition", "wall_s", "lat_mean_s", "lat_sd",
            "lat_p95", "lat_max_s", "server_proc_mean_s",
            "gpu_peak_mem_MiB", "gpu_mean_util_pct",
            "host_ram_peak_GiB", "files_per_min", "realtime_factor"]
    hdr = ["conc", "rep", "wall", "lat_mean", "lat_sd", "lat_p95", "lat_max",
           "proc_mean", "gpu_peak", "util%", "ram_GiB", "files/min", "RTF"]
    widths = [max(len(h), 8) for h in hdr]
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(w) for c, w in zip(cols, widths)))

    _print_across_repetitions(rows)
    _print_stage_profile(rows)

    # sizing implication, from the single-job (concurrency=1) throughput
    base = next((r for r in rows if r["concurrency"] == 1), rows[0])
    tp = base.get("files_per_min")
    print("\nInterpretation")
    print("-" * 100)
    if isinstance(tp, (int, float)) and tp:
        print(f"* One server instance sustains ~{tp:.2f} files/min for this audio "
              f"({base.get('realtime_factor','?')}x real-time).")
        # Scaling efficiency against the level-1 baseline, reported rather than
        # asserted: the old text always blamed the GIL, which is wrong once the
        # load is spread over replica processes.
        peak = max(
            (r for r in rows
             if isinstance(r.get("files_per_min"), (int, float))),
            key=lambda r: r["files_per_min"], default=None)
        if peak and peak["concurrency"] > 1:
            speedup = peak["files_per_min"] / tp
            eff = 100 * speedup / peak["concurrency"]
            print(f"* Peak throughput {peak['files_per_min']:.2f} files/min at "
                  f"concurrency {peak['concurrency']} = {speedup:.2f}x over one "
                  f"job in flight ({eff:.0f}% of linear).")
            print(f"* Below ~100% of linear, concurrent jobs are contending. The "
                  f"stage profile above localizes it: a stage whose mean grows "
                  f"in step with concurrency is serialized, one that stays flat "
                  f"is genuinely overlapping.")
        print(f"* Peak GPU memory ≈ {base.get('gpu_peak_mem_MiB','?')} MiB "
              f"(+{base.get('gpu_mem_over_idle_MiB','?')} MiB over idle at "
              f"concurrency 1). Memory is rarely the binding constraint here; "
              f"GPU compute is, so spreading replicas over --gpus buys more "
              f"than stacking them on one.")
    print("* Quote a level's mean only alongside its sd/cv; with --repeat > 1 the "
          "across-repetitions table shows whether that mean is reproducible.")


if __name__ == "__main__":
    main()
