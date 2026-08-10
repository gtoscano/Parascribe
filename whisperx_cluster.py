#!/usr/bin/env python3
"""
Launch WhisperX replicas across one or more GPUs behind an nginx gateway that
round-robins requests — a single endpoint that spreads load, with no client-side
URL juggling.

Why replicas (not threads): the pipeline (diarization, alignment,
assign_word_speakers) is largely pure-Python CPU work that holds the GIL, so
multiple worker *threads* in one process serialize on the GIL and only contend
for the GPU. Separate *processes* each get their own interpreter and CUDA
context, so they truly overlap — up to GPU-compute saturation (~2-3 replicas per
GPU is the sweet spot; memory is not the limit at ~13 GB each).

Multiple GPUs: pass --gpus with several indices. Jobs never cross a GPU (no model
sharding, no inter-GPU traffic), so throughput scales ~linearly with GPU count.
Replicas are interleaved across the GPUs, so even a small batch fans out across
all of them. The single gateway fronts every replica regardless of GPU, and the
job-id prefix routing is unchanged.

How routing works: each replica stamps its REPLICA_ID onto every job_id
("<id>-<token>"). nginx round-robins POST /v1/transcribe across the pool, and
routes GET /v1/jobs/<job_id> back to the owning replica by the numeric prefix —
necessary because each replica's job registry is in-memory and per-process.

Layout:
  gateway (nginx)  -> host GATEWAY_PORT (default 8357)   <- point clients here
  replica i        -> host BASE_PORT + i (default 8360)  <- direct access / benchmarks
  total replicas   = --replicas (per GPU)  x  number of --gpus

Usage:
  # 3 replicas on one GPU (unchanged single-GPU behaviour)
  REPLICAS=3 python3 whisperx_cluster.py up

  # 3 replicas per GPU across 4 GPUs = 12 replicas
  python3 whisperx_cluster.py up --replicas 3 --gpus 0,1,2,3

  python3 whisperx_cluster.py gen                  # just (re)write compose + nginx
  python3 whisperx_cluster.py ps                   # status
  python3 whisperx_cluster.py urls                 # print the gateway URL
  python3 whisperx_cluster.py urls --direct        # direct replica URLs (CSV)
  python3 whisperx_cluster.py down                 # stop + remove everything

Env vars (override defaults):
  REPLICAS      replicas PER GPU              (default 2)
  WHISPERX_GPUS comma-separated GPU indices   (default "1"; falls back to WHISPERX_GPU)
  GATEWAY_PORT  host port for the gateway     (default 8357)
  BASE_PORT     first direct replica port     (default 8360)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD_CONTEXT = "./whisperx"
IMAGE = "whisperx-api:local"
GEN_FILE = HERE / "docker-compose-whisperx.cluster.yml"
NGINX_FILE = HERE / "whisperx-nginx.conf"
NETWORK = "whisperx-net"
PROJECT = "whisperx-cluster"  # compose project name, isolates these containers


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _default_gpus() -> str:
    return os.environ.get("WHISPERX_GPUS") or os.environ.get("WHISPERX_GPU") or "1"


def parse_gpus(spec: str):
    """'0,1,2,3' -> ['0','1','2','3']."""
    gpus = [g.strip() for g in str(spec).split(",") if g.strip() != ""]
    return gpus or ["1"]


def gpu_for(i: int, gpus) -> str:
    """Interleave replicas across GPUs so small batches fan out across all of
    them (replica 0->gpu[0], 1->gpu[1], ...)."""
    return gpus[i % len(gpus)]


def total_replicas(per_gpu: int, gpus) -> int:
    return per_gpu * len(gpus)


# Rough resident footprint of one active replica (measured ~13 GB); used only
# for a soft "may not fit" advisory, not a hard limit.
ESTIMATED_REPLICA_MIB = 13000


def _nvidia_smi_gpus():
    """Return {index(str): free_MiB(int)} from nvidia-smi, or None if it cannot
    be queried (binary missing, no driver, etc.)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    gpus = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] != "":
            try:
                gpus[parts[0]] = int(parts[1])
            except ValueError:
                gpus[parts[0]] = None
    return gpus or None


def verify_gpus(per_gpu: int, gpus) -> bool:
    """Guardrail before `up`: confirm every requested GPU exists, and warn if a
    GPU likely lacks free memory for its replicas. Returns False only when a
    requested GPU is genuinely absent."""
    present = _nvidia_smi_gpus()
    if present is None:
        print("! could not run nvidia-smi — skipping GPU verification "
              "(use --no-gpu-check to silence).")
        return True
    missing = [g for g in gpus if g not in present]
    if missing:
        print(f"! requested GPU(s) {missing} not found. "
              f"Available: {sorted(present)}.")
        print("  Fix --gpus (or WHISPERX_GPUS), or pass --no-gpu-check to override.")
        return False
    # Soft memory advisory (does not block).
    need = per_gpu * ESTIMATED_REPLICA_MIB
    for g in gpus:
        free = present.get(g)
        if free is not None and free < need:
            print(f"! GPU {g}: ~{free} MiB free, but {per_gpu} replica(s) may "
                  f"need ~{need} MiB — risk of out-of-memory.")
    print(f"GPU check OK: {sorted(set(gpus))} present "
          f"({len(gpus)} GPU(s), {total_replicas(per_gpu, gpus)} replicas).")
    return True


def _replica_block(idx: int, host_port: int, gpu: str) -> str:
    """One replica service. WORKERS=1, a REPLICA_ID for gateway routing, pinned
    to physical GPU `gpu` (remapped to cuda:0 inside the container)."""
    return f"""  whisperx-{idx}:
    image: {IMAGE}
    container_name: whisperx-{idx}
    ports:
      - "{host_port}:8000"
    dns:
      - 1.1.1.1
      - 8.8.8.8
    ipc: host
    shm_size: '16gb'
    ulimits:
      memlock: -1
      stack: 67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["{gpu}"]
              capabilities: [gpu, compute, utility]
    environment:
      - HUGGINGFACE_HUB_TOKEN=${{HUGGINGFACE_HUB_TOKEN}}
      - HF_TOKEN=${{HUGGINGFACE_HUB_TOKEN}}
      - HF_HOME=/workspace/.cache/huggingface
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - CUDA_VISIBLE_DEVICES=0
      - WHISPER_MODEL=large-v3
      - COMPUTE_TYPE=float16
      - DEFAULT_LANGUAGE=en
      - BATCH_SIZE=16
      - DATA_DIR=/data
      # One worker per replica: parallelism comes from multiple replica
      # processes, not threads. REPLICA_ID is stamped onto job_ids so the
      # gateway can route job lookups back here.
      - WORKERS=1
      - REPLICA_ID={idx}
    volumes:
      - ~/.cache/huggingface:/workspace/.cache/huggingface
      - ./whisperx/data:/data
    networks:
      - {NETWORK}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
"""


def _gateway_block(gateway_port: int, total: int) -> str:
    depends = "\n".join(
        f"      whisperx-{i}:\n        condition: service_healthy"
        for i in range(total))
    return f"""  whisperx-gateway:
    image: nginx:alpine
    container_name: whisperx-gateway
    ports:
      - "{gateway_port}:8000"
    volumes:
      - ./{NGINX_FILE.name}:/etc/nginx/nginx.conf:ro
    depends_on:
{depends}
    networks:
      - {NETWORK}
    restart: unless-stopped
    healthcheck:
      # busybox wget (nginx:alpine) needs split flags (not bundled -qO-) and
      # 127.0.0.1 (it resolves localhost to IPv6 ::1, which nginx doesn't bind).
      test: ["CMD-SHELL", "wget -q -O - http://127.0.0.1:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
"""


def gateway_url(gateway_port: int) -> str:
    return f"http://localhost:{gateway_port}"


def replica_urls(total: int, base_port: int):
    return [f"http://localhost:{base_port + i}" for i in range(total)]


def gpu_map(per_gpu: int, gpus):
    """{gpu -> [replica indices]} for display."""
    m = {g: [] for g in gpus}
    for i in range(total_replicas(per_gpu, gpus)):
        m[gpu_for(i, gpus)].append(i)
    return m


def generate_compose(per_gpu: int, base_port: int, gateway_port: int,
                     gpus) -> str:
    total = total_replicas(per_gpu, gpus)
    header = (
        "# AUTO-GENERATED by whisperx_cluster.py — do not edit by hand.\n"
        f"# {total} replica(s) = {per_gpu}/GPU x {len(gpus)} GPU(s) "
        f"{gpus}, behind an nginx gateway.\n"
        f"# gateway -> :{gateway_port} (clients)   replicas -> "
        f":{base_port}..{base_port + total - 1} (direct)\n"
        "# Regenerate:  python3 whisperx_cluster.py gen\n"
        f"# Build the image first:  docker build -t {IMAGE} {BUILD_CONTEXT}\n"
        "networks:\n"
        f"  {NETWORK}:\n"
        "    driver: bridge\n"
        "services:\n"
    )
    replicas_yaml = "".join(
        _replica_block(i, base_port + i, gpu_for(i, gpus)) for i in range(total)
    )
    return header + replicas_yaml + _gateway_block(gateway_port, total)


def generate_nginx(total: int) -> str:
    upstream = "\n".join(
        f"        server whisperx-{i}:8000 max_fails=0;" for i in range(total)
    )
    return f"""# AUTO-GENERATED by whisperx_cluster.py — do not edit by hand.
# One worker so round-robin state is global and distribution is even (each
# nginx worker keeps its own RR counter; multiple workers skew small samples).
# A single worker easily handles this proxy's request rate.
worker_processes 1;
events {{ worker_connections 1024; }}
http {{
    # Docker's embedded DNS, so `whisperx-<n>` resolves at request time.
    resolver 127.0.0.11 valid=10s ipv6=off;
    access_log /dev/stdout;
    error_log  /dev/stderr;

    upstream whisperx_pool {{
{upstream}
    }}

    client_max_body_size 0;        # allow large audio uploads
    proxy_http_version 1.1;
    proxy_connect_timeout 30s;
    proxy_read_timeout 7200s;      # long transcriptions
    proxy_send_timeout 7200s;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

    server {{
        listen 8000;

        # Job status/results -> the replica that owns the job (numeric prefix
        # in the id, e.g. /v1/jobs/2-abc... -> whisperx-2). Variable upstream
        # is resolved via the embedded DNS above.
        location ~ ^/v1/jobs/(?<rid>[0-9]+)- {{
            set $backend whisperx-$rid:8000;
            proxy_pass http://$backend;
        }}

        # Submit + health + models + job list -> round-robin the pool.
        location / {{
            proxy_pass http://whisperx_pool;
        }}
    }}
}}
"""


def _print_layout(per_gpu: int, base_port: int, gateway_port: int, gpus,
                  prefix: str = "") -> None:
    total = total_replicas(per_gpu, gpus)
    print(f"{prefix}{total} replica(s) = {per_gpu}/GPU x {len(gpus)} GPU(s) {gpus}")
    for g, idxs in gpu_map(per_gpu, gpus).items():
        names = ", ".join(f"whisperx-{i}" for i in idxs)
        print(f"  GPU {g}: {names}")
    print(f"  gateway  -> {gateway_url(gateway_port)}")
    print(f"  direct   -> :{base_port}..{base_port + total - 1}")


def write_files(per_gpu: int, base_port: int, gateway_port: int, gpus) -> None:
    total = total_replicas(per_gpu, gpus)
    GEN_FILE.write_text(generate_compose(per_gpu, base_port, gateway_port, gpus))
    NGINX_FILE.write_text(generate_nginx(total))
    _print_layout(per_gpu, base_port, gateway_port, gpus,
                  prefix=f"Wrote {GEN_FILE.name} + {NGINX_FILE.name}: ")


def _run(cmd: list, **kw) -> int:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=HERE, **kw).returncode


def _compose(*args) -> list:
    return ["docker", "compose", "-p", PROJECT, "-f", str(GEN_FILE), *args]


def _ensure_files(per_gpu, base_port, gateway_port, gpus):
    if not GEN_FILE.exists() or not NGINX_FILE.exists():
        write_files(per_gpu, base_port, gateway_port, gpus)


def cmd_up(per_gpu, base_port, gateway_port, gpus,
           no_check=False, dry_run=False) -> int:
    if not no_check and not verify_gpus(per_gpu, gpus):
        return 2
    if dry_run:
        print("\n[dry-run] plan (nothing written, no containers started):")
        _print_layout(per_gpu, base_port, gateway_port, gpus)
        print("[dry-run] re-run without --dry-run to build and start.")
        return 0
    write_files(per_gpu, base_port, gateway_port, gpus)
    rc = _run(["docker", "build", "-t", IMAGE, BUILD_CONTEXT])
    if rc != 0:
        return rc
    rc = _run(_compose("up", "-d", "--remove-orphans"))
    if rc != 0:
        return rc
    total = total_replicas(per_gpu, gpus)
    gw = gateway_url(gateway_port)
    directs = replica_urls(total, base_port)
    print(f"\nGateway (round-robins across {total} replicas on "
          f"{len(gpus)} GPU(s)):\n  {gw}")
    print("Point the client at the gateway — one endpoint, load-balanced:")
    print(f"  python3 transcribe.py *.m4a --url {gw} --concurrency {total}")
    print(f"  (or: export WHISPERX_URL={gw})")
    print("\nDirect replica URLs (for benchmarking individual replicas):")
    print(f"  {','.join(directs)}")
    return 0


def cmd_down(per_gpu, base_port, gateway_port, gpus) -> int:
    _ensure_files(per_gpu, base_port, gateway_port, gpus)
    return _run(_compose("down"))


def cmd_ps(per_gpu, base_port, gateway_port, gpus) -> int:
    _ensure_files(per_gpu, base_port, gateway_port, gpus)
    return _run(_compose("ps"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Manage WhisperX replicas across GPUs behind an nginx gateway.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("action", choices=["up", "down", "gen", "ps", "urls"],
                    help="up=build+start, down=stop, gen=write files only, "
                         "ps=status, urls=print gateway URL (--direct for replicas)")
    ap.add_argument("--replicas", type=int, default=_env_int("REPLICAS", 2),
                    help="replicas PER GPU (env REPLICAS, default 2); "
                         "total = replicas x number of --gpus")
    ap.add_argument("--gpus", default=_default_gpus(),
                    help="comma-separated GPU indices (env WHISPERX_GPUS / "
                         "WHISPERX_GPU, default '1'), e.g. 0,1,2,3")
    ap.add_argument("--base-port", type=int, default=_env_int("BASE_PORT", 8360),
                    help="first direct replica host port (env BASE_PORT, default 8360)")
    ap.add_argument("--gateway-port", type=int,
                    default=_env_int("GATEWAY_PORT", 8357),
                    help="nginx gateway host port (env GATEWAY_PORT, default 8357)")
    ap.add_argument("--direct", action="store_true",
                    help="for `urls`: print direct replica URLs instead of the gateway")
    ap.add_argument("--dry-run", action="store_true",
                    help="for `up`: verify GPUs and print the plan, without "
                         "writing files, building images, or starting containers")
    ap.add_argument("--no-gpu-check", action="store_true",
                    help="for `up`: skip the nvidia-smi GPU-existence check")
    args = ap.parse_args()

    if args.replicas < 1:
        ap.error("--replicas must be >= 1")
    gpus = parse_gpus(args.gpus)
    total = total_replicas(args.replicas, gpus)

    a = (args.replicas, args.base_port, args.gateway_port, gpus)

    if args.action == "urls":
        if args.direct:
            print(",".join(replica_urls(total, args.base_port)))
        else:
            print(gateway_url(args.gateway_port))
        return 0
    if args.action == "gen":
        write_files(*a)
        return 0
    if args.action == "up":
        return cmd_up(*a, no_check=args.no_gpu_check, dry_run=args.dry_run)
    if args.action == "down":
        return cmd_down(*a)
    if args.action == "ps":
        return cmd_ps(*a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
