# Resume handoff

Updated: 2026-10-26

## Current session: Parascribe v2 (read this first)

The user requested saving this session before closing it and plans to restore
it with `/resume`. Read this handoff and `../MEMORY.md`, then inspect git state.
The earlier publication handoff below is historical; do not treat its old
runtime/dirty-worktree descriptions as current or resume benchmark work by
default. Do not automatically change branches, restart services, or push files.

### Repository and branch verified on 2026-10-26

- Working clone: `/home/gtoscano/Parascribe`, owner account `gtoscano`, previously
  accessed as `gtoscano@gpuccino`.
- Current branch: `feature/v2-session-voiceprints`.
- HEAD: `df17b69` (`Update weights to identify speakers`).
- `main`: `9e2f71c` (`State the non-monotonic throughput prediction up front`).
- Feature branch is six commits ahead of main. With rename detection at 20%,
  the branch diff is 29 files, 3,513 insertions, 241 deletions. Default rename
  detection counts the extended server as a new/deleted file and inflates this
  to 30 files, 4,033 insertions, 761 deletions.
- Before this handoff edit, worktree was clean and HEAD matched the locally
  cached `origin/feature/v2-session-voiceprints`. No remote fetch was performed
  in the branch-comparison/save turns; do not claim fresh remote verification.
- This save modifies this handoff and `AGENTS.md`; it does not commit or push.

Feature commits, oldest first:

1. `3f5aea3` Add Parakeet realtime sessions and scoped voiceprints.
2. `e3b08f2` Add local Cohere comparison and generic Parascribe service naming.
3. `a47fbd5` Add browser tester for v2 sessions and model comparison.
4. `c2d660a` Version 2.
5. `e922fa1` add filess.
6. `df17b69` Update weights to identify speakers.

### Last user question and answer

The user saw this in another terminal:

```text
connorsg@gpus:~/Parascribe$ git branch
  agent/import-paper-source
* main
```

Explained that this is a different account/clone, and possibly a different host;
do not assume `gpus` and `gpuccino` are the same machine. Each clone checks out
its own branch. Suggested read-only checks `git status`, `git remote -v`, then
`git fetch origin` and `git branch -a` in that other clone; if its remote is the
same and the feature branch exists remotely, use
`git switch --track origin/feature/v2-session-voiceprints` (or `git switch
feature/v2-session-voiceprints` if already local). No commands were run in the
other account and no branch switches were performed. Preserve any local work.

### Current application design

- Same project with versioned APIs, not a separate replacement project.
- V1 batch WhisperX/Whisper pipeline remains present.
- V2 sentence and microphone transcription uses Parakeet TDT 0.6B v3 INT8 ONNX
  on CPU and/or local Cohere Transcribe 03-2026 Q8 through transcribe.cpp.
  Modes: `parakeet`, `cohere`, `compare`; Cohere requires an explicit language.
  Compare retains both results/timings, primary session text is Parakeet.
- Voice embedding model is still `pyannote/wespeaker-voxceleb-resnet34-LM`.
  ReDimNet2 was researched and compared, NOT integrated. B3 was recommended as
  a candidate for a later benchmark, not established as a measured improvement.
- Voiceprints and transcripts are persisted in SQLite by conversation/session;
  never globally match people across sessions. Deletion disables capture while
  preserving transcripts; ending may delete signatures. Backups need separate
  retention handling. Voiceprints are not authentication.
- Latest speaker-matching commit uses duration-weighted centroids, matching and
  updating thresholds of 0.30 by default, and new-speaker confirmation enabled:
  unmatched voices are pending until a second matching utterance confirms them;
  the previous pending utterance is then relabeled. Pending embeddings are also
  removed by voiceprint deletion. Inspect README-v2/code for calibration context.
- Browser tester `/v2/test`, sources `parascribe/web/`; supports uploads,
  microphone, manual commits, energy-based automatic sentence endpointing, and
  live provisional snapshot re-decoding. Live is NOT native token streaming or
  grammatical sentence detection. Automatic chunks cap at 20 seconds, manual
  chunks at 55 seconds; drafts request decoding about every 1.5 seconds.
- Drafts are ephemeral, do not create transcript entries or voiceprints, and
  are invalidated on commit/discard. Final utterances perform speaker matching.
- In-page tabs: Live transcript and Final & comparison, not a new window.
  Prominent final speaker labels; live draft identity remains pending. User
  display names are browser-local, session-scoped annotations exported as
  `speaker_names`; not server-side verified names.
- The app does not yet implement per-user session authorization. SSH accounts
  authenticate tunnels, not ownership/isolation of API sessions. Classroom
  testing should not assume sessions are confidential from other users.

### Known unresolved issue and future scope

- Empty primary transcript still raises `ValueError("no speech was transcribed
  from this utterance")` in `run_session_utterance`; confirmed in code on
  2026-10-26. User saw a raw traceback. This has NOT been fixed.
- Earlier diagnosis (2026-10-11): failed job `0-aad1adaa3a30` had 6.064 seconds
  of audio and empty Parakeet output; previous chunks succeeded. Audio was
  deleted, so silence versus a missed speech segment could not be established.
- Discussed gracefully reporting no recognized speech and continuing listening,
  rather than surfacing a traceback. Do not claim implemented.
- Server-side speech endpointing/continuous automatic finalization is NOT
  implemented. Browser detects pauses and sends commits; server buffers audio
  and runs explicitly requested preview/final decoding.
- No authorization to implement ReDimNet2, rerun benchmarks, or expand scope
  was inferred from model-comparison questions.

### Access, testing, and verification boundaries

- User explicitly said DO NOT change Tailscale configuration.
- Previously deployed gateway port 8357, one replica on GPU 1/direct port 18360;
  8360/8361 were occupied by other services. Remote upload URL:
  `http://gpuccino:8357/v2/test`. Tailscale IP then was `100.127.117.17`.
  These runtime/network details were not rechecked during this save turn.
- Microphone requires localhost or HTTPS. Each student runs this on their own
  computer (Mac Terminal works), using their own server account:

  ```bash
  ssh -N -L 18357:127.0.0.1:8357 YOUR_USERNAME@gpuccino
  ```

  Keep the tunnel open and browse `http://localhost:18357/v2/test`.
  Do not use the owner's username in student-facing instructions.
- Full test instructions: `../TO_TEST.md`; API/design details: `../README-v2.md`.
- Earlier implementation validation passed 20 unit/API/model tests and a real
  Chromium smoke test with synthetic microphone audio for manual/automatic/live
  modes, comparisons, names, export and deletion. Those runs predate the latest
  speaker-matching commit; do not claim they newly validate HEAD. No tests were
  rerun during the 2026-10-26 branch comparison/save work.
- Preserve existing work, other applications, model caches and datasets. Shell
  calls in the sandbox currently fail with `bwrap: loopback: Failed RTM_NEWADDR:
  Operation not permitted`; use the normal approval/escalation mechanism.

## Historical publication handoff (2026-08-10)

The content below is preserved for publication continuity. Its runtime and
worktree statements describe that earlier date, NOT the current v2 deployment.

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
