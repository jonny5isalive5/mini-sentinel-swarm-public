# Historical development record

The original Git objects, branches and tags are not included. The following is a redacted text record of earlier development; original commit identifiers are provenance, not public checkout targets.

# Development history of Flat Swarm V1

The commits below are the genuine development history of `uai32-swarm/` in `the private origin repository`, extracted with
`git subtree split --prefix=uai32-swarm` from the frozen head `530ec02` (pull request #4,
PRIVATE_ORIGIN, branch `claude/loving-turing-5z6uex`). The tree of the last
extracted commit is byte-identical to that directory at `530ec02`. Every commit was authored by Claude (Fable 5.1)
in a Claude Code cloud session on 8 October 2026; the historical messages are reproduced with private session references redacted because they record what
was wrong at each step. The commit after these (the first one unique to this repository) only moves the files into
`flat-swarm-v1/`, vendors the µAI-32 v1.0.0 binary the code runs against, and adds the provenance documents.

| original (private origin) | historical extracted SHA | date | author | subject |
|---|---|---|---|---|
| `530ec02` (private historical reference) | `788169c` | 2026-10-08 | Claude | uai32-swarm: close the auditor's cosmetic residuals after retest |
| `3f03e26` (private historical reference) | `606788a` | 2026-10-08 | Claude | uai32-swarm: remediated Stage 6 (budget-matched monoliths), final determination |
| `e2d7485` (private historical reference) | `0b04e4d` | 2026-10-08 | Claude | uai32-swarm: remediation after the independent audit (zero BLOCKERs) |
| `2c2a316` (private historical reference) | `08e0aaa` | 2026-10-08 | Claude | uai32-swarm: clean Stage 6 results, final determination, restored evidence |
| `982ea33` (private historical reference) | `09ae521` | 2026-10-08 | Claude | uai32-swarm: delete stale model files before creating baselines (reproducibility) |
| `d713352` (private historical reference) | `43fc341` | 2026-10-08 | Claude | uai32-swarm: chunked router (memory flaw found at N=1000), final determination draft |
| `4f9b679` (private historical reference) | `0d01650` | 2026-10-08 | Claude | uai32-swarm: Mini Sentinel specialist swarm, stages 1-5, 7, 8 and spawning |

## Historical commit messages (private session and unrelated-project references redacted)

### `4f9b679` (2026-10-08) uai32-swarm: Mini Sentinel specialist swarm, stages 1-5, 7, 8 and spawning

```
A thin Python coordination layer over the unmodified released uai32
v1.0.0 binary (every prediction and weight update is made by that
binary on a 576-byte model file).  Synthetic task world, canonical
brain + byte-copy clones, file-backed specialist library with metadata
(hash, lineage, familiarity model, held-out competence), familiarity
router with spread penalty and "unsupported" decisions, weighted-vote
arbitration that exposes disagreement, two-specialist composition,
resource measurements (RSS via a C launcher), populations, learning
over time (isolation, forgetting, retrain, replace, clone, branch,
remove) and expert spawning.  Results as JSON per stage; FAILURES.md
keeps what went wrong and what is weak.  Stage 6 (scale to 1,000) and
the final determination follow.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `d713352` (2026-10-08) uai32-swarm: chunked router (memory flaw found at N=1000), final determination draft

```
The vectorised familiarity router materialised a rows x specialists x
features tensor; at 1,000 specialists the orchestrator peaked at 7.9 GB
while accuracy was unaffected.  Routing is now done in chunks of 256
rows.  The unchunked run is preserved under results/ as evidence, and
FAILURES.md records the flaw, the fix and the growing orchestration
overhead.  FINAL.md drafts the determination against the ten success
criteria and answers the final question with the measured scale table;
its Stage 6 numbers are refreshed from the chunked rerun that follows.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `982ea33` (2026-10-08) uai32-swarm: delete stale model files before creating baselines (reproducibility)

```
uai32 train continues an existing model file, so monolith and compound
baselines trained in an earlier run were trained further in the next
one (equal-storage monolith at N=2: 81.0%, 83.0%, 79.0% across three
runs while every swarm number was identical).  create() now deletes any
existing file and run_all.sh starts from an empty work/.  Quick stages
rerun from scratch; the clean Stage 6 run is in progress and its results
replace the in-progress file in the next commit.  FAILURES.md item 12.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `2c2a316` (2026-10-08) uai32-swarm: clean Stage 6 results, final determination, restored evidence

```
Stage 6 rerun from an empty work/ with the chunked router: swarm 84.0 /
83.2 / 85.1 / 84.3% at N = 2 / 10 / 100 / 1,000 against 81.0 / 65.8 /
31.8 / 25.9% for the same-size monolith and 81.0 / 79.2 / 56.7 / 31.3%
(capped at 2,048 hidden units) for the equal-storage monolith; routing
97.8 to 100%; orchestrator 38 MB to 487 MB.  Stage 2 rerun is identical
to the byte.  FINAL.md carries the final numbers; the two superseded
Stage 6 runs are restored as evidence; run_all.sh no longer deletes
them.  Root README updated (unrelated-project reference omitted).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `e2d7485` (2026-10-08) uai32-swarm: remediation after the independent audit (zero BLOCKERs)

```
Audit findings (eleven IMPORTANT, six LATER) addressed in one pass:
monolith baselines get the specialists' training budget and record
it; "equal storage" labelled as model-file storage with metadata
excluded, model-only swarm bytes reported; routing, pure uai32
subprocess and orchestration times recorded separately; specialist
working set corrected to float32 parameters (1,300 B); routing MACs
count two operations per feature; failed invocations counted; a
specialist failing its hash check is not reported as selected;
per-stage library names and per-level canonical files; temporary row
files deleted; no timestamps in history.  Added a composition baseline
with the swarm's own supervision (one brain on device states, then the
table): 56.0% vs 76.4%.  Wording: what numpy fits, "domain identity
explicit in the input" qualifiers, exact numbers.  FAILURES.md items
11-13 record the audit and the remediation.  Quick stages rerun from
clean; the remediated Stage 6 run is in progress.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `3f03e26` (2026-10-08) uai32-swarm: remediated Stage 6 (budget-matched monoliths), final determination

```
Stage 6 rerun with monoliths trained on the specialists' budget: swarm
84.0 / 83.2 / 85.1 / 84.3% at N = 2 / 10 / 100 / 1,000 vs 80.0 / 70.0 /
33.5 / 26.0% (same-size monolith) and 81.0 / 79.0 / 63.5 / 30.2% (equal
model storage; N = 1,000 capped at 2,048 units, 20 epochs).  Routing
2 to 112 us/row, uai32 subprocess 0.03 ms/row, Python 0.02 ms/row;
MACs/query 284 to 20,244 vs 529 to 30,724.  FINAL.md carries the
corrected table, the auditor's budget-matched N = 10 result, the
composition baseline with the swarm's own supervision, and the capped
comparison stated plainly.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

### `530ec02` (2026-10-08) uai32-swarm: close the auditor's cosmetic residuals after retest

```
Retest in a fresh copy: 14 of 17 findings fixed, 3 partially fixed with
cosmetic residue only.  Fixed here: README routing-cost sentence and
metadata history fields, 7,896,960 KB stated as 7.5 GiB, the RSS probe's
temporary row file deleted.  No measurement affected; FAILURES.md item 14
records the retest.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: PRIVATE_SESSION_REDACTED
```

## Failure and audit record

What went wrong, what was superseded and what the independent audit found are in `flat-swarm-v1/FAILURES.md`
(items 1 to 14) and in the superseded result files kept on purpose in `flat-swarm-v1/results/`
(`stage6_superseded_old_router.*`, `stage6_superseded_unchunked_router.*`). None of it has been removed.
