#!/usr/bin/env python3
"""
Client for the WhisperX API + optional summarization via your vLLM Qwen3 server.

Handles one file or many. Two ways to spread many files across replicas:

  * Gateway (recommended): point --url at the nginx gateway (default :8357) and
    use --concurrency to keep several jobs in flight; the gateway round-robins
    each to a replica. One endpoint, no URL juggling. See whisperx_cluster.py.
  * Direct pool (advanced): pass --urls with the replicas' direct URLs; files
    are round-robined and each replica runs one at a time (bypasses the gateway,
    handy for benchmarking individual replicas).

Examples
--------
# One file, auto speaker count, save transcript next to the audio:
  python transcribe.py meeting.m4a

# You know there are exactly 3 speakers:
  python transcribe.py meeting.m4a --num-speakers 3

# Many files via the gateway, 3 in flight (matches a 3-replica cluster):
  python transcribe.py *.m4a --num-speakers 8 --concurrency 3

# Many files straight at the replica pool (bypass the gateway):
  python transcribe.py *.m4a --urls http://localhost:8360,http://localhost:8361,http://localhost:8362

# Transcribe AND summarize with Qwen3 (vLLM on port 8355):
  python transcribe.py meeting.m4a --summarize

# Just summarize an existing transcript file:
  python transcribe.py --summarize-file transcript.txt

Endpoints (override with env or flags):
  WHISPERX_URL   default http://localhost:8357   (nginx gateway; round-robins internally)
  WHISPERX_URLS  comma-separated replica pool for direct round-robin; overrides WHISPERX_URL
  VLLM_URL       default http://localhost:8355   (your Qwen3 general model)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

WHISPERX_URL = os.environ.get("WHISPERX_URL", "http://localhost:8357")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8355")

SUMMARY_PROMPT = """You are summarizing a transcript of a spoken conversation with
multiple speakers (labeled Speaker 1, Speaker 2, ...). Produce:

1. **Overview** — 2-3 sentences on what the conversation was about.
2. **Key points** — bullet list of the most important points.
3. **Decisions** — any decisions made (or "none").
4. **Action items** — who committed to what (reference speakers), or "none".

Transcript:
---
{transcript}
---
"""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _new_stats(source_path, operation, base_url=None):
    source_path = Path(source_path)
    stats = {
        "schema_version": 1,
        "operation": operation,
        "source_file": str(source_path),
        "status": "running",
        "started_at": _utc_now(),
        "client": {"durations_s": {}},
    }
    if source_path.exists():
        stats["source_bytes"] = source_path.stat().st_size
    if base_url:
        stats["server_url"] = base_url
    return stats


def _stats_path(source_path):
    return Path(f"{source_path}.timings.json")


def _save_stats(source_path, stats):
    out = _stats_path(source_path)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return out


def _capture_server_stats(stats, job):
    stats["job_id"] = job.get("job_id", stats.get("job_id"))
    stats["server"] = {
        "timestamps": {
            key: job[key]
            for key in ("received_at", "uploaded_at", "created_at", "queued_at",
                        "started_at", "finished_at")
            if job.get(key)
        },
        "durations_s": dict(job.get("timings", {})),
    }


def transcribe(audio_path, base_url, language="en", diarize=True,
               num_speakers=None, min_speakers=None, max_speakers=None,
               poll=5, timeout=7200, tag="", stats=None):
    """Transcribe one file and add client/server timings to ``stats``."""
    pfx = f"[{tag}] " if tag else ""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    stats = stats if stats is not None else _new_stats(
        audio_path, "transcribe", base_url)
    durations = stats.setdefault("client", {}).setdefault("durations_s", {})
    transcription_started = time.perf_counter()
    wait_started = None
    polls = 0
    try:
        print(f"{pfx}→ Uploading {audio_path.name} to {base_url} ...")
        data = {"language": language, "diarize": str(diarize).lower()}
        for k, v in (("num_speakers", num_speakers),
                     ("min_speakers", min_speakers),
                     ("max_speakers", max_speakers)):
            if v:
                data[k] = v

        upload_started = time.perf_counter()
        try:
            with audio_path.open("rb") as f:
                response = requests.post(
                    f"{base_url}/v1/transcribe",
                    files={"file": (audio_path.name, f)}, data=data)
        finally:
            durations["upload_http_s"] = round(
                time.perf_counter() - upload_started, 6)
        response.raise_for_status()
        job_id = response.json()["job_id"]
        stats["job_id"] = job_id
        print(f"{pfx}  job_id = {job_id}")

        wait_started = time.perf_counter()
        last_stage = None
        while True:
            status_started = time.perf_counter()
            response = requests.get(f"{base_url}/v1/jobs/{job_id}")
            durations["status_requests_s"] = round(
                durations.get("status_requests_s", 0.0)
                + time.perf_counter() - status_started, 6)
            polls += 1
            response.raise_for_status()
            job = response.json()
            _capture_server_stats(stats, job)
            status, stage = job.get("status"), job.get("stage")
            if stage != last_stage:
                print(f"{pfx}  [{int(time.perf_counter()-wait_started)}s] "
                      f"{status} / {stage}")
                last_stage = stage
            if status == "done":
                break
            if status == "error":
                raise RuntimeError("Transcription failed:\n"
                                   + job.get("error", "unknown error"))
            if time.perf_counter() - wait_started > timeout:
                raise TimeoutError("Timed out waiting for transcription.")
            time.sleep(poll)
        durations["wait_until_done_s"] = round(
            time.perf_counter() - wait_started, 6)
        stats["client"]["status_requests"] = polls

        download_started = time.perf_counter()
        try:
            response = requests.get(
                f"{base_url}/v1/jobs/{job_id}/transcript.txt")
            response.raise_for_status()
            text = response.text
        finally:
            durations["transcript_download_s"] = round(
                time.perf_counter() - download_started, 6)

        out_txt = audio_path.with_suffix(".transcript.txt")
        write_started = time.perf_counter()
        try:
            out_txt.write_text(text)
        finally:
            durations["transcript_write_s"] = round(
                time.perf_counter() - write_started, 6)
        print(f"{pfx}✓ Transcript saved: {out_txt}  "
              f"({job.get('num_speakers', '?')} speakers detected)")
        return text, out_txt
    finally:
        if wait_started is not None:
            durations["wait_until_terminal_s"] = round(
                time.perf_counter() - wait_started, 6)
            stats["client"]["status_requests"] = polls
        durations["transcription_total_s"] = round(
            time.perf_counter() - transcription_started, 6)


def summarize(transcript, vllm_url, max_tokens=2048, tag="", stats=None):
    pfx = f"[{tag}] " if tag else ""
    durations = (stats if stats is not None else {}).setdefault(
        "client", {}).setdefault("durations_s", {})
    summarize_started = time.perf_counter()
    try:
        model_started = time.perf_counter()
        try:
            response = requests.get(f"{vllm_url}/v1/models")
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        finally:
            durations["summary_model_lookup_s"] = round(
                time.perf_counter() - model_started, 6)
        if stats is not None:
            stats["summary_model"] = model
            stats["summary_server_url"] = vllm_url
        print(f"{pfx}→ Summarizing with {model} at {vllm_url} ...")
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": SUMMARY_PROMPT.format(transcript=transcript)}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        inference_started = time.perf_counter()
        try:
            response = requests.post(
                f"{vllm_url}/v1/chat/completions", json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        finally:
            durations["summary_inference_http_s"] = round(
                time.perf_counter() - inference_started, 6)
    finally:
        durations["summarization_total_s"] = round(
            time.perf_counter() - summarize_started, 6)


def process_one(audio, base_url, args, tag="", print_summary=False):
    """Full pipeline for one file; always save a timing sidecar."""
    pfx = f"[{tag}] " if tag else ""
    stats = _new_stats(
        audio, "transcribe_and_summarize" if args.summarize else "transcribe",
        base_url)
    overall_started = time.perf_counter()
    try:
        text, out_txt = transcribe(
            audio, base_url, language=args.language,
            diarize=not args.no_diarize, num_speakers=args.num_speakers,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            tag=tag, stats=stats)
        summary_out = None
        if args.summarize:
            summary = summarize(text, args.vllm_url, tag=tag, stats=stats)
            summary_out = Path(audio).with_suffix(".summary.md")
            write_started = time.perf_counter()
            try:
                summary_out.write_text(summary)
            finally:
                stats["client"]["durations_s"]["summary_write_s"] = round(
                    time.perf_counter() - write_started, 6)
            if print_summary:
                print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
                print(summary)
            print(f"{pfx}✓ Summary saved: {summary_out}")
        stats["status"] = "done"
        return audio, out_txt, summary_out
    except Exception as exc:
        stats["status"] = "error"
        stats["error"] = str(exc)
        raise
    finally:
        stats["finished_at"] = _utc_now()
        stats["client"]["durations_s"]["overall_s"] = round(
            time.perf_counter() - overall_started, 6)
        timing_out = _save_stats(audio, stats)
        print(f"{pfx}  Timings saved: {timing_out}")


def resolve_urls(args):
    """Replica pool: --urls (or WHISPERX_URLS) wins, else the single --url."""
    if args.urls:
        return [u.strip().rstrip("/") for u in args.urls.split(",") if u.strip()]
    return [args.url.rstrip("/")]


def run_batch_gateway(files, url, concurrency, args):
    """Fire files at a single endpoint (the nginx gateway) with up to
    `concurrency` in flight; the gateway round-robins each to a replica and
    routes its status polls back by the job_id prefix."""
    conc = max(1, concurrency)
    print(f"Submitting {len(files)} file(s) to {url}, up to {conc} in flight "
          f"(gateway round-robins to replicas)\n")
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(process_one, f, url, args, Path(f).stem): f
                for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append((str(f), str(e)))
                print(f"[{Path(f).stem}] ! failed: {e}")
    return results, errors


def run_batch_direct(files, urls, args):
    """Round-robin files across the replicas' direct URLs; each replica
    processes its own files sequentially (one at a time), all in parallel."""
    groups = [[] for _ in urls]
    for i, f in enumerate(files):
        groups[i % len(urls)].append(f)
    plan = ", ".join(f"{u.rsplit(':', 1)[-1]}→{len(g)}"
                     for u, g in zip(urls, groups) if g)
    print(f"Distributing {len(files)} file(s) across {len(urls)} replica(s) "
          f"[port→files]: {plan}\n")

    results, errors = [], []

    def replica_worker(url, gfiles):
        out = []
        for f in gfiles:
            tag = Path(f).stem
            try:
                out.append(process_one(f, url, args, tag=tag))
            except Exception as e:
                errors.append((str(f), str(e)))
                print(f"[{tag}] ! failed: {e}")
        return out

    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        futs = [ex.submit(replica_worker, u, g)
                for u, g in zip(urls, groups) if g]
        for fut in as_completed(futs):
            results.extend(fut.result())
    return results, errors


def main():
    ap = argparse.ArgumentParser(description="WhisperX transcribe + Qwen3 summarize")
    ap.add_argument("audio", nargs="*", help="audio/video file(s) to transcribe")
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip speaker identification")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--summarize", action="store_true",
                    help="also summarize via the vLLM Qwen3 server")
    ap.add_argument("--summarize-file",
                    help="skip transcription; summarize this existing text file")
    ap.add_argument("--url", default=WHISPERX_URL,
                    help=f"WhisperX server URL (default {WHISPERX_URL}; "
                         "also settable via WHISPERX_URL)")
    ap.add_argument("--urls", default=os.environ.get("WHISPERX_URLS"),
                    help="comma-separated replica pool for direct round-robin "
                         "(env WHISPERX_URLS); overrides --url and bypasses the gateway")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="max jobs in flight when submitting many files to the "
                         "gateway (--url); set to the replica count (default 3). "
                         "Ignored in --urls direct mode.")
    ap.add_argument("--vllm-url", default=VLLM_URL,
                    help=f"vLLM server URL for --summarize (default {VLLM_URL})")
    args = ap.parse_args()

    urls = resolve_urls(args)

    # Summarize-only path (no transcription).
    if args.summarize_file:
        stats = _new_stats(args.summarize_file, "summarize")
        overall_started = time.perf_counter()
        try:
            text = Path(args.summarize_file).read_text()
            summary = summarize(text, args.vllm_url, stats=stats)
            out = Path(args.summarize_file).with_suffix(".summary.md")
            write_started = time.perf_counter()
            try:
                out.write_text(summary)
            finally:
                stats["client"]["durations_s"]["summary_write_s"] = round(
                    time.perf_counter() - write_started, 6)
            stats["status"] = "done"
            print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
            print(summary)
            print(f"\n✓ Summary saved: {out}")
        except Exception as exc:
            stats["status"] = "error"
            stats["error"] = str(exc)
            raise
        finally:
            stats["finished_at"] = _utc_now()
            stats["client"]["durations_s"]["overall_s"] = round(
                time.perf_counter() - overall_started, 6)
            timing_out = _save_stats(args.summarize_file, stats)
            print(f"  Timings saved: {timing_out}")
        return

    if not args.audio:
        ap.error("provide at least one audio file (or use --summarize-file)")

    files = args.audio
    if len(files) == 1:
        # Single file → one replica, clean untagged output (unchanged UX).
        if len(urls) > 1:
            print(f"(1 file, {len(urls)}-replica pool → using {urls[0]}; "
                  "round-robin kicks in with multiple files)")
        try:
            process_one(files[0], urls[0], args, tag="", print_summary=True)
        except Exception as e:
            sys.exit(str(e))
        return

    # Batch: direct pool (--urls) → per-replica round-robin;
    # otherwise a single endpoint (the gateway) → concurrency-capped fan-out.
    if len(urls) > 1:
        results, errors = run_batch_direct(files, urls, args)
    else:
        results, errors = run_batch_gateway(files, urls[0], args.concurrency, args)
    print(f"\n{'=' * 60}\nDone: {len(results)}/{len(files)} transcribed"
          + (f", {len(errors)} failed" if errors else "")
          + f"  (pool of {len(urls)})")
    for f, e in errors:
        first = e.splitlines()[0] if e else "?"
        print(f"  ! {f}: {first}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
