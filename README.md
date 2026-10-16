# Mini Sentinel Swarm

This publication contains the established **Flat Swarm V1** baseline and its recorded failures. It is a project-only snapshot with a fresh Git history. Unreleased research branches remain private and are not presented as completed experiments here. See [publication notes](PUBLICATION_NOTES.md) and [provenance](PUBLICATION_PROVENANCE.json).

Experiments on composing many independently trained copies of one tiny trainable brain, **Mini Sentinel**
(µAI-32 v1.0.0: a 9,188-byte executable plus a 576-byte model file), into a more capable system through
routing, arbitration and composition, at a resource cost that keeps the brain's advantage.

| component | status |
|---|---|
| `flat-swarm-v1/` | **Flat Swarm V1 baseline.** Computational files and numerical results match the private original at `530ec02`; three log path headers are redacted. The untouched frozen baseline remains private. |
| `uai32/` | the vendored v1.0.0 brain binary and its source, used unmodified |
| `PROVENANCE.md` | research attribution, historical origin identifiers, audit scope and failure records |
| `HISTORY.md` | a redacted text account of development, with historical commit identifiers; no original Git objects |

## What Flat Swarm V1 established

On a synthetic workload of N unrelated domains (each a different nonlinear 4-class rule, domain identity
explicit in the input), 576-byte specialists routed by familiarity hold about 84% accuracy from 2 to 1,000
domains while a same-size monolith falls to 26% and an equal-model-storage monolith to 30 to 64%; routing
accuracy 97.8 to 100%; 94 to 100% of out-of-distribution inputs refused; two specialists composed through a
table reach 76% on a task no single brain reads, against 56% for one brain with the same supervision. Its
limitations are part of the baseline and are not to be broadened: domain identity is explicit in the synthetic
task; routing cannot tell same-domain experts apart under noise, edge or adversarial conditions; forgetting
inside a specialist is total; spawning clusters crudely; the Python orchestrator costs 39 to 489 MB and routing
grows to 112 µs per row at 1,000 specialists. Full determination: `flat-swarm-v1/FINAL.md`; measurements:
`flat-swarm-v1/results/REPORT.md`; failures: `flat-swarm-v1/FAILURES.md`.

## Running the baseline

```sh
cd flat-swarm-v1 && ./run_all.sh      # python3 + numpy and a C compiler; about 15 minutes; seeded and reproducible
```

The code finds the brain at `../uai32/dist/uai32`; `UAI32=...` overrides it.

## Verification and scope

Run `python -B verify_publication.py` at the repository root to check publication hashes without running experiments. All superseded result files and the failure record remain included. The earlier run command is supplied for future reproduction only; preparation of this copy did not execute it.

The vendored brain retains its [Apache-2.0 licence](uai32/LICENSE) and [NOTICE](uai32/NOTICE). No new distribution licence is asserted for the surrounding Swarm code. Publication does not grant additional third-party rights.
