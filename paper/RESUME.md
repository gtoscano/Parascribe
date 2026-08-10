# Resume handoff

Updated: 2026-08-10

Canonical repository: `/home/gtoscano/Parascribe`

The paper and source code were copied here from
`/home/gtoscano/LLMServer` on 2026-08-10. Use the Parascribe clone for future
publication, commit, and push work. The original LLMServer tree remains the
runtime deployment and benchmark source; do not delete or modify it merely
because this copy exists.

## User objective

Prepare the paper in this directory for arXiv. The user previously requested:

1. Review coherence and writing style.
2. Correct the batch model.
3. Add repetitions and variability.

Items 2 and 3 have been implemented. When the user says `resume`, do not rerun
the benchmark campaign automatically. First inspect the current worktree and ask
or infer which remaining publication task the user wants next.

## Completed manuscript work

- Corrected the batch model using
  `q = floor(N/R)`, `r = N mod R`, and
  `W = q*s_R` when `r = 0`, otherwise `W = q*s_R + s_r`.
- Corrected the thread experiment description: `WORKERS=2` is fixed and rows
  vary the number of in-flight jobs (1, 2, and 4).
- Added repetitions:
  - five trials per thread, process-scaling, and gateway configuration;
  - three trials per ten-file batch configuration.
- Added mean and sample standard deviation to the methodology and tables.
- Added sample-SD error bars to the process-scaling and utilization figures.
- Updated the abstract, results, discussion, threats to validity, multi-GPU
  extrapolation, conclusion, and reproduction appendix.
- Added a machine-readable aggregate:
  `measurements/repetitions/summary.csv`.

## Important repeated results

- All 36 retained trial CSVs had zero failed jobs.
- One stream:
  - W-short: 5.94 files/min; GPU utilization 43.4 +/- 2.1%.
  - W-long: 0.99 files/min; GPU utilization 43.8 +/- 0.4%.
- Two threads in one process:
  - one active worker: 5.936 +/- 0.005 files/min;
  - two active workers: 4.246 +/- 0.005 files/min;
  - throughput regression: 28.5%.
- Independent processes:
  - W-short, 3 replicas: 8.890 +/- 0.019 files/min, 1.50x speedup.
  - W-long, 3 replicas: 1.540 files/min, 1.56x speedup.
  - W-long, 4 replicas: 1.684 +/- 0.009 files/min, 1.70x speedup.
- Gateway at 3 replicas:
  - W-short: 8.994 +/- 0.434 files/min versus 8.890 direct.
  - W-long: 1.574 +/- 0.013 files/min versus 1.540 direct.
  - No throughput penalty was observed; the comparison is descriptive, not an
    equivalence test.
- Ten-file batches:
  - W-short measured 60.0 +/- 2.3 s; model predicted 62.5 +/- 0.5 s.
  - W-long measured 370.9 +/- 15.0 s; model predicted 395.0 +/- 2.0 s.
  - The repeated data invalidate the former 0.5% accuracy claim. The synchronized
    wave estimate is conservative here, overpredicting the repeated means by
    4.0% and 6.1%. The paper explains desynchronized queued stages as the likely
    structural reason.

All `+/-` values above are sample standard deviations.

## Files and verification

- Main source: `main.tex`
- Built paper: `main.pdf`
- Raw trials: `measurements/repetitions/*.csv`
- Aggregate: `measurements/repetitions/summary.csv`
- Build command: `make` (two pdflatex passes)
- Last build: successful, 11 pages.
- Last log audit: no unresolved references, LaTeX errors, overfull boxes, or
  underfull boxes.
- Structured PDF text for the revised result pages was checked successfully.

## Runtime state

- The temporary `WORKERS=2` benchmark container was removed.
- The original deployment was restored:
  - three WhisperX process replicas on GPU 1;
  - gateway at `http://localhost:8357`;
  - direct replicas at ports 8360--8362.
- The gateway health endpoint was verified healthy with `workers: 1`.
- Do not change or stop this deployment unless the next user request requires it.

## Worktree cautions

- In this new clone, most copied source files are initially untracked and
  `README.md` is modified relative to the minimal remote repository.
- `/home/gtoscano/LLMServer/paper.tar.gz` was deliberately not copied.
- Preserve unrelated user changes.
- The normal filesystem patch helper previously failed with:
  `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
  The successful fallback was the system `apply_patch` command invoked with
  escalated execution and the patch supplied on stdin.

## Sensible next publication tasks

Only perform these after the user chooses or clearly requests them:

- Complete a final coherence/style edit of the revised prose.
- Verify arXiv source packaging and regenerate an archive without build junk.
- Review title/abstract positioning and claims for appropriate caution.
- Commit or publish the changes if explicitly requested.
