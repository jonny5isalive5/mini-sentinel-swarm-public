# Preserved failure evidence and corrections

Things that did not work, were wrong on the first attempt, or are weaker than hoped. Kept on purpose.

1. **Peak-RSS measurement was wrong twice.** Measuring the uai32 child's RSS through Python's `subprocess`
   or `os.posix_spawn` reported about 50 MB for every command, including `/bin/true`, because the kernel
   charges a vfork-style child with its parent's pages at exec time. Fixed by spawning through
   `tools/maxrss.c`, a tiny C launcher: specialist process 2,244 KB against a 1,376 KB floor for `/bin/true`.
   The earlier numbers were discarded, not reported.
2. **The first router always chose the "noisy" expert.** Ranking candidates by distance-over-radius favours any
   specialist whose training inputs were widely spread, because its radius is large. In Stage 7 the noisy expert
   was selected for 294 to 299 of 300 inputs under every condition. Fixed by ranking with a diagonal-Gaussian
   score that charges for training spread (`0.5·mean(z²) + mean(log spread)`). The first Stage 6 run, made with
   the old ranking, was stopped at N = 1,000 and its partial output is kept as
   `results/stage6_superseded_old_router.json` / `.log` (routing accuracy there was 100% / 97.8% / 99.0% at
   N = 2 / 10 / 100; the support decision was unchanged by the fix).
3. **Input statistics cannot tell most conditions apart (Stage 7).** After the fix, the swarm detects the
   *shifted* condition (254 of 300 inputs to the shifted expert) but spreads normal, noisy, edge and adversarial
   inputs almost uniformly over the five experts, because those conditions leave the input distribution nearly
   unchanged. k = 3 arbitration recovers oracle-level accuracy in four of five conditions; under the *edge*
   condition every expert is below 50% and so is the swarm. The swarm cannot identify the best expert for noisy,
   edge or adversarial inputs from the inputs alone; this is a real limit, not a bug.
4. **Catastrophic forgetting is total.** A specialist continued-trained on a second domain for 40 epochs drops
   from 89% to 33% on its original domain (chance is about 25 to 35%). uai32 has no replay or regularisation;
   the swarm's protection against forgetting is isolation between files, not anything inside a brain.
5. **Composition baseline was first under-trained.** The single-brain baselines for the compound task were
   first trained on 3,000 rows for 40 epochs; they were given 5,000 rows (the same number the ten specialists
   saw in total) and 100 epochs before the reported run. They stayed at 40 to 45% against the swarm's 76%.
6. **Spawning clusters crudely.** The stretch goal commissions a new specialist from the buffered unsupported
   inputs by taking the points within 0.1 of the buffer's mean identifying coordinates, falling back to the
   neighbourhood of the first buffered point. With two unseen domains arriving together it spawned three
   specialists, one of them trained on a mixed cluster (domain 10 ends at 60%, domain 11 at 88%). A proper
   clustering step is a later improvement.
7. **Specialist accuracy is 83 to 85%, not 95%+.** The random teachers are harder than a 10-16-4 network
   trained with 500 rows and 60 epochs can fit exactly. This is the same brain everywhere, so comparisons are
   fair, but absolute numbers should be read with that in mind. A first version with 400 rows / 40 epochs
   reached 81 to 83%; the defaults were raised once, before any comparison was run.
8. **The orchestrator is not small.** Python + numpy sit at about 50 MB RSS and the per-query cost is
   dominated by starting a uai32 process (about 1.4 ms). The specialists themselves are 576 bytes and run in
   2.2 MB of process memory with an 816-byte working set. The honest total-RAM figure for this implementation
   is the Python figure; a C orchestrator is a later improvement, not an assumption used in any claim.
9. **Equal-storage monolith capped.** At N = 1,000 the equal-storage monolith would need about 19,000 hidden
   units (580 KB); training it with per-sample SGD on 400,000 rows was not feasible in the time available, so
   the comparison uses a 2,048-unit monolith (about 62 KB, roughly the storage of 107 specialists). The cap is
   reported in the table.
10. **The router did not scale in memory (found at N = 1,000, Stage 6).** The vectorised familiarity computation
    formed a rows × specialists × features tensor for the whole 50,000-row evaluation set: 50,000 × 1,000 × 10
    doubles, several times over, and the orchestrator's peak RSS reached 7,896,960 KB (7.5 GiB) while accuracy
    was unaffected (84.3%, routing 98.7%). At N = 100 the same code needed 124 MB and at N = 10 42 MB, so the
    flaw was invisible below 1,000. Fixed by routing in chunks of 256 rows (peak temporary memory
    256 × N × 10 doubles). The unchunked run is preserved as `results/stage6_superseded_unchunked_router.json`
    / `.log`; the reported Stage 6 numbers come from the rerun with the chunked router.
11. **Timing labels were swapped until the audit.** `decide()` started its clock after `route()`, so the value
    reported as "invocation" contained the per-row Python bookkeeping and the value reported as
    "orchestration" was only the routing tensor. The 0.84 ms/row "orchestration" at N = 1,000 before the memory
    fix was the unchunked router. Since the remediation the JSON records routing time, pure uai32 subprocess time
    (`Sentinel.seconds`) and the Python remainder separately; the conclusion that coordination, not the
    specialists, dominates at N = 1,000 stands, and the numbers in FINAL.md are the corrected ones.
12. **Baseline models were not reproducible between runs.** `uai32 train` continues training when the model
    file already exists, and `Sentinel.create()` did not delete a stale file first, so monolith and compound
    baselines trained in an earlier run (sometimes by an earlier version of the code) were trained further in the
    next one: the equal-storage monolith at N = 2 scored 81.0%, 83.0% and 79.0% in three consecutive runs. The
    audit showed the swarm was affected too: the stale `canonical_s6.model` was continued instead of recreated per
    level, so swarm task accuracy drifted slightly across runs (N = 10: 83.6%, 83.6%, 83.2%; N = 100: 84.9%,
    84.9%, 85.1%) while routing accuracy stayed identical. Found by comparing `results/stage6_superseded_*.json`.
    Fixed by deleting the file in `create()`, per-level canonical files, and starting `run_all.sh` from an empty
    `work/`; the fixed code is reproducible to the byte (auditor: two clean runs identical apart from timings).
13. **Independent audit (one pass, zero BLOCKERs, eleven IMPORTANT, six LATER) and the single remediation.**
    Fixed in code: monolith baselines now get the specialists' budget (500 rows per domain, 60 epochs; the
    capped N = 1,000 equal-storage monolith 20 epochs, marked) and record rows/epochs; "equal storage" is
    labelled as model-file storage with metadata excluded, and the swarm's model-only bytes are reported next
    to its total; routing, pure subprocess and orchestration times are recorded separately; the specialist
    working set is 4 bytes per float32 parameter plus activations plus a row (1,300 B, not 816 B); routing
    MACs count two operations per feature per specialist; failed invocations are counted, not assumed;
    a specialist that fails its hash check is no longer reported as "selected"; per-stage library names and
    per-level canonical files; temporary row files are deleted; history has no timestamps; continued training
    updates history. Added: a composition baseline with the swarm's own supervision (one brain trained on all
    device states, applied to both halves, then the table): 56.0% against the swarm's 76.4%. Reworded: "no
    learning algorithm" to the precise statement of what numpy fits; criteria 4 and 9 qualified with "domain
    identity explicit in the input"; numbers that did not trace exactly. Recorded, not fixed: the storage column
    still counts metadata JSON that duplicates index.json; the one-brain composition baseline is the only
    budget-matched baseline at Stage 4. The auditor's own numbers for the fair N = 10 comparison (monolith 83.0
    to 84.6% against the swarm's 84.2%) are quoted in FINAL.md as the auditor's measurement.
14. **Retest after remediation (auditor, fresh copy, HEAD 3f03e26).** 14 of 17 findings FIXED, 3 PARTIALLY
    FIXED with only cosmetic residue (a README sentence still counting routing as N × 10 multiply-adds, a stale
    `t` field in the metadata table, "7.9 GB" for 7,896,960 KB, and four temporary row files left by the RSS
    probe); every rechecked number reproduced from a clean state and traced to the JSON. Those four items are
    fixed in the closing commit without rerunning the experiment (none affects a measurement).
