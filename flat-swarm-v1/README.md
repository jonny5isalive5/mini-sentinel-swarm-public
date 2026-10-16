# Mini Sentinel specialist swarm

An experiment on whether many independently trained copies of one tiny brain, coordinated by a lightweight
layer, give useful broad capability at a resource cost far below one monolithic model. The brain is the
released µAI-32 v1.0.0 binary (`../uai32/dist/uai32`, 9,188 bytes, SHA-256 `996d73cb…`, tag `uai32-v1.0.0`),
unmodified. Every class prediction and every weight update in this directory is made by that binary on a
576-byte model file. The coordination layer (`swarm.py`, Python + numpy) fits 21 unsupervised statistics per
specialist (centroid, spread and radius of its training inputs) and applies fixed formulas (Gaussian familiarity
score, vote weight, disagreement margin, composition table, spawn-cluster threshold): a label-free
nearest-centroid router, not a classifier of the task. No larger model is used anywhere, at build time or at
run time; the synthetic task generator in the same file is the "test generator" the challenge allows.

Measured results are in `results/REPORT.md` (generated from `results/*.json`); the final determination is in
`FINAL.md`; what did not work is in `FAILURES.md`.

## Architecture

```
                 canonical brain (576 B uai32 model: normalisation stats + random weights, 0 epochs)
                                   | byte copy, byte copy, byte copy ...
      +----------------------------+----------------------------+---------------------+
      v                            v                            v                     v
  specialist s0000            specialist s0001            specialist s0002   ...   spawned_N
  (trained on domain 0)       (trained on domain 1)       (domain 2, branch)        (buffered niche)
  s0000.model + s0000.json    s0001.model + s0001.json    ...                       library/ on disk,
                                                                                    nothing resident
                                        ^ one uai32 process per invocation, loads one file, exits
                                        |
  input row ---> route() ---> decide() ---+---> answer + provenance record
                 |                        |
                 | familiarity:           +--> k=1: the best-ranked specialist whose region contains the input
                 | index.json only        +--> k>1: arbitration by weighted vote; disagreement exposed
                 | (centroid, spread,     +--> no specialist's region contains the input: "unsupported"
                 |  radius per specialist)+--> corrupt file (hash mismatch): that specialist cannot contribute
                 |
  compound input ---> split ---> route/decide each half ---> compose (table) ---> answer   (Stage 4)
  unsupported inputs + external labels ---> buffer ---> clone canonical, train, register   (stretch)
```

**Brain.** 10 inputs (4 identify the domain, 6 describe the condition), 16 hidden units, 4 classes: 576 bytes.
**Canonical brain.** Created once from inputs spread over the whole input space with zero epochs, so every
specialist shares the same input normalisation and the same random starting weights. Clones are byte copies.
**Specialist.** A clone continued-trained (`uai32 train` on the existing file) on one domain under one
condition, plus a metadata record. Training, use and restart are all separate processes reading the file.

## Specialist metadata (`library/<id>.json`, aggregated in `library/index.json`)

| field | meaning |
|---|---|
| `id`, `version` | name; version increments on retrain/replace |
| `sha256`, `bytes` | hash and size of the model file; `verify()` refuses a specialist whose file no longer matches |
| `parent`, `parent_sha256` | what it was cloned from (`canonical` or another specialist) |
| `trained_on` | `{domain, condition, rows, epochs, lr, seed}` |
| `centroid[10]`, `spread[10]`, `radius` | familiarity model: mean and standard deviation of its training inputs, and the 99th percentile of the standardised distance of those inputs |
| `val_acc` | held-out accuracy on 100 rows of its own training distribution (its historical competence) |
| `history` | list of `{event, sha256}`: train, retrain, replace, clone, branch, spawn, continue |

The router needs only `index.json` (about 1 KB per specialist, of which 21 numbers are used for routing); model files are opened only by the uai32
process that is about to use them.

## Router

For an input `x` and every specialist `s`: `z = (x - centroid_s) / spread_s`, relative distance
`rel = sqrt(mean(z²)) / radius_s` (inside the familiar region when ≤ 1), and a diagonal-Gaussian score
`0.5·mean(z²) + mean(log spread_s)` that charges a specialist for a broad training distribution (without that
term the expert trained on noisy data wins every input because its radius is the largest). Candidates are the
specialists whose region contains `x`, ranked by the score. Cost: two operations per feature per specialist (2 × 10 × N) plus one forward pass, numpy
vectorised: about 1 µs per row at N = 10 and 15 µs at N = 100. No specialist's own confidence
is used for routing, only for arbitration. Routing at N = 1,000 costs about 300 µs per row in numpy.

## Arbitration (k > 1)

The top-k candidates inside their regions are invoked; each returns a class and its softmax confidence
(three decimals, as printed by uai32). Vote weight = `val_acc × confidence × exp(-rel)`. Weights are summed
per class; the winner answers. If the runner-up class holds at least 75% of the winner's weight, the record
is marked `disagreement` and both sides are exposed with their supporters. No consensus is forced.

## Composition (Stage 4)

A compound input is two device readings (20 numbers). No specialist can read it (they take 10). The
composition layer splits it, routes each half, and combines the two device states through a fixed 4×4 table
(the layer knows the table, not the device rules). The baselines are single brains trained directly on the
compound task with the same number of rows and more epochs.

## Persistence

A specialist is a file. "Restart" in every stage means a fresh uai32 process reading that file; outputs are
compared across processes and hashes are compared before and after use. Continued training rewrites the file
and its hash; nothing else on disk changes (checked by hashing every other specialist).

## Running it

```sh
./run_all.sh                       # all stages with seed 1, then results/REPORT.md (about 10 minutes)
python3 swarm.py 1 2 3             # individual stages; 6 takes --sizes 2,10,100,1000
python3 report.py
```

Needs python3 + numpy, a C compiler (for `tools/maxrss.c`, the RSS probe) and the released binary. All randomness
is seeded; rerunning reproduces the JSON files apart from timings.

## What is measured where

| stage | file | proves |
|---|---|---|
| 1 | `results/stage1.json` | identical start, divergence by experience, same-experience control, persistence |
| 2 | `stage2.json`, `stage2_provenance.jsonl` | bounded competences, routing, provenance, unsupported inputs |
| 3 | `stage3.json` | arbitration, exposed disagreement |
| 4 | `stage4.json` | composition beats any single brain on a task no specialist can read |
| 5 | `stage5.json` | per-specialist size, index size, load time, routing cost, process RSS, working set |
| 6 | `stage6.json` | 2 → 10 → 100 → 1,000 specialists against same-size and equal-storage monoliths |
| 7 | `stage7.json` | populations: which expert the swarm picks as conditions change |
| 8 | `stage8.json` | continued training, forgetting, retrain, replace, clone, branch, remove |
| stretch | `stretch.json` | spawning a specialist for an unsupported niche |

## Honest limits of the setup

- Tasks are synthetic: random nonlinear teachers over a 6-dimensional condition space, one per domain, with
  domains identified by four noisy coordinates (centres at least 0.15 apart, noise 0.03: five standard
  deviations). Domain identity is therefore explicit in the input and routing is a trivially separable
  nearest-centroid problem; the hard part of the experiment is the capacity comparison, not the routing. Real
  workloads may not partition this cleanly.
- The orchestrator is Python and spawns one process per specialist batch; its own memory (about 50 MB,
  numpy) and the ~1.4 ms process start dominate the per-query cost. A C orchestrator would remove most of it;
  the measured numbers are reported as they are.
- Confidence comes from uai32's three-decimal softmax, which saturates; arbitration weights are therefore coarse.
- Labels for spawning come from the environment (the synthetic teacher), i.e. external supervision.
