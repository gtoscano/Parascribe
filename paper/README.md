# Paper: Throughput Scaling of a GPU Speech Transcription and Diarization Service

LaTeX source for an arXiv-style measurement paper describing the WhisperX
scaling work in this repository: why thread-level parallelism regresses
throughput (GIL/CPU-bound pipeline), why process-level replicas scale it, the
nginx gateway with job-id-prefix routing, and a conservative remainder-aware
batch-sizing model evaluated across repeated trials.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | The complete paper. **Self-contained** — bibliography is inline (`thebibliography`), and all figures are generated inline with `pgfplots` (no external image files). |
| `Makefile` | Build targets (`make`, `make docker`, `make clean`). |
| `figures/` | Empty; kept for optional externalized figures. |

## Before submitting

- **Authors:** Gregorio Toscano (corresponding) and Grace Connors, both at The
  Catholic University of America. Update affiliations/order if needed before
  submission.
- **arXiv category:** the natural primary category is **cs.DC** (Distributed,
  Parallel, and Cluster Computing), with **cs.SD** (Sound) or **eess.AS** as a
  cross-list.
- **Numbers:** every figure in the paper is a measured value from this session's
  benchmarks on a single NVIDIA RTX PRO 6000. Re-run `benchmark_transcribe.py`
  if you change hardware or workloads.

## Building

You need a TeX Live installation (the `pgfplots`, `booktabs`, `hyperref`,
`listings`, `subcaption` packages — all in a standard full install). No `bibtex`
run is required because the bibliography is inline.

### Option A — local TeX Live
```bash
make            # runs pdflatex twice (resolves references) -> main.pdf
```
or manually:
```bash
pdflatex main.tex && pdflatex main.tex
```

### Option B — Docker (no local TeX needed)
```bash
make docker     # uses the texlive/texlive image to build main.pdf
```
which is equivalent to:
```bash
docker run --rm -v "$PWD:/w" -w /w texlive/texlive \
    sh -c 'pdflatex main.tex && pdflatex main.tex'
```
> The `texlive/texlive` image is large (~4 GB) on first pull.

### Option C — Overleaf / arXiv
Upload `main.tex` directly. Overleaf and arXiv's build system compile it as-is;
set the compiler to **pdfLaTeX**. arXiv accepts the single `.tex` file (figures
are inline pgfplots, so nothing else needs to be uploaded).

## arXiv packaging

arXiv wants the source, not just the PDF. Because this project is self-contained,
a submission tarball is simply:
```bash
tar czf arxiv-submission.tar.gz main.tex
```
Upload that tarball; arXiv compiles it. (If you later add external figure files,
include them in the tarball.)
