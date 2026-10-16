# Final determination

*Numbers below are from `results/REPORT.md` (seed 1, gcc 13.3 / glibc 2.39 x86-64, released uai32 v1.0.0),
after the independent audit and the single remediation pass (FAILURES.md items 10 to 13). Monolith baselines
are trained with the same 500 rows per domain and 60 epochs as each specialist, except the capped N = 1,000
equal-storage monolith (20 epochs). "Equal storage" means N × the 576-byte model file; metadata is excluded.*

## What was proven

| success criterion | evidence | verdict |
|---|---|---|
| 1. Two identical brains diverge through different experiences | Stage 1: one SHA-256 at start; different after training on different domains; a third clone given the same experience stays byte-identical; each clone scores 84 to 87% on its own domain and 18 to 27% on the other's; the two answer differently on 83% of inputs | proven |
| 2. Learning persists | Stage 1: fresh processes reproduce every output from the files; hashes unchanged by use; Stage 8: hashes change only when a specialist is trained | proven |
| 3. Specialists coexist independently | Stage 8: training one specialist leaves the other nine files byte-identical; a corrupted file is caught by its hash and excluded; removal deletes one file | proven (isolation is structural: one file per brain) |
| 4. A lightweight router selects useful specialists | Stages 2 and 6: routing accuracy 97.8 to 100% from 2 to 1,000 specialists using only per-specialist centroid, spread and radius; 21 numbers of routing data per specialist; no model file opened to route | proven, on a task whose domain identity is explicit in the input (wrong-domain selections are below 0.3%; the 0 to 2.2% routing errors are radius refusals) |
| 5. Multiple specialists contribute to one task | Stage 3: weighted vote of three same-domain experts raises accuracy from 82.0% to 86.9% (normal inputs); Stage 4: two specialists composed through a table reach 76.4% on a task no single specialist can read; the table is an oracle given to the composition layer and the specialists had device-state (intermediate) supervision, so the fair comparison is the one-brain baseline with the same supervision and table (see below) | proven |
| 6. Disagreement is exposed or arbitrated | Stage 3: 3.3% (normal) and 8.2% (edge) of inputs flagged as disagreements with both sides and their supporters recorded; accuracy on flagged inputs is 46 to 47%, so the flag is informative | proven |
| 7. The system says when no suitable expert exists | Stage 2: 100% of inputs from three never-seen domains and 100% of uniform noise marked unsupported; Stage 6: 94 to 100% of out-of-distribution inputs unsupported at every N, 0 to 2.2% of in-domain inputs wrongly refused | proven |
| 8. Add, replace, retrain, clone, remove independently | Stage 8: all six operations performed and verified by hash and accuracy; branching one expert into two gives three distinct hashes and condition-specific behaviour | proven |
| 9. Scales materially beyond a handful | Stage 6: 1,000 specialists, 2.6 MB of storage (576 KB of model files), task accuracy 84.3%, routing 98.7%, every invocation counted and none failed; run-to-run reproducible | proven on a task whose domain identity is explicit in the input, with one flaw found and fixed (router memory) |
| 10. Total resource cost keeps the Mini Sentinel advantage | per specialist 576 B model + about 1 KB metadata (21 numbers used for routing); a specialist runs in a 2.2 MB process (1.4 MB of which is glibc floor) with a 1,300-byte working set (float32 parameters, activations, one row); 1,000 specialists cost 20,244 multiply-adds per query including routing, against 30,724 for the capped monolith | proven for the specialists; the Python orchestrator is the expensive part (see boundary) |

## What remains unproven or failed

- **Choosing among experts for the same domain by condition.** Input statistics detect a distribution shift
  (254 of 300 shifted inputs routed to the shifted expert) but cannot separate normal, noisy, edge and adversarial
  inputs; the swarm reaches oracle accuracy only through k = 3 arbitration, and under edge conditions every
  expert and the swarm are below 50% (Stage 7).
- **Forgetting inside a specialist is total** (89% → 33% after learning a second domain). The swarm has no
  mechanism against it other than isolation and retraining from the canonical brain (Stage 8).
- **Spawning works but clusters crudely**: two unseen domains arriving together produced three specialists,
  one of them mixed; the niche ended at 60% and 88% accuracy (stretch).
- **The equal-storage comparison at N = 1,000 is capped.** The fair monolith would need about 19,000 hidden
  units; the trained one has 2,048 (62 KB, the storage of about 107 specialists) trained for 20 epochs and
  reaches 30.2%. At N = 100, where the fair monolith (1,916 units, 57.6 KB, 60 epochs) was trained, it reaches
  63.5% against the swarm's 85.1%. Storage comparisons exclude the swarm's metadata (about 4.5 × its model bytes).
- **Real workloads were not tested.** All domains are synthetic random teachers chosen to partition cleanly.

## The final question

*Can a large population of tiny, separately trained specialists provide useful broad capability through
routing and composition at substantially lower resource cost than a monolithic model for the same constrained
workload?*

**Yes, for workloads that partition into recognisable domains, and the boundary is the coordination layer,
not the specialists.**

Quantified, on this workload (N domains, each a different nonlinear 4-class rule on 6 condition features):

| N | swarm accuracy | same-size monolith (576 B, same budget) | equal model-storage monolith (same budget) | swarm model bytes / with metadata | swarm MACs/query (routing + one forward) | monolith MACs/query |
|---|---|---|---|---|---|---|
| 2 | 84.0% | 80.0% | 81.0% (NH 35, 1.1 KB) | 1.2 KB / 5.1 KB | 284 | 529 |
| 10 | 83.2% | 70.0% | 79.0% (NH 188, 5.7 KB) | 5.8 KB / 26 KB | 444 | 2,824 |
| 100 | 85.1% | 33.5% | 63.5% (NH 1,916, 57.6 KB) | 58 KB / 257 KB | 2,244 | 28,744 |
| 1,000 | 84.3% | 26.0% | 30.2% (NH 2,048, 61.5 KB, capped, 20 epochs) | 576 KB / 2.6 MB | 20,244 | 30,724 |

The auditor's own budget-matched runs at N = 10 gave the equal-storage monolith 83.0% (NH 189) and a monolith
sized to the swarm's *total* storage including metadata (NH 869, 26 KB) 84.6%, against the swarm's 84.2% on
its test draw: at ten domains a fairly trained monolith is as good as the swarm. At N = 100 the auditor's
budget-matched monolith reached 65.5% against 85.4%.

- The swarm's accuracy is flat in N; both monoliths fall with N. At 100 domains the swarm beats a monolith of
  the same model storage and training budget by 22 points while using 13 times fewer multiply-adds per query,
  and beats the same-size monolith by 52 points. At 1,000 domains the comparison is capped (a fair monolith
  would need about 19,000 hidden units); the trend from 10 to 100 is the evidence.
- Composition adds capability no single brain has: 76% versus at most 45% for brains trained directly on the
  compound task from composed labels, and versus 56% for one brain given the swarm's own supervision (trained
  on every domain's device states, applied to both halves, composed through the same oracle table).
- The cost that does not stay small is coordination: the Python orchestrator uses 39 MB at N = 2 and 489 MB
  at N = 1,000 (after the router was chunked; it needed 7,896,960 KB, 7.5 GiB, before), and routing costs 2 µs per row at
  N = 10 and 112 µs at N = 1,000, against 0.03 ms of uai32 subprocess time and 0.02 ms of Python bookkeeping
  per row that do not grow with N. At 1,000 specialists routing is the largest per-row cost. The specialists
  keep the Mini Sentinel advantage; the orchestrator must be engineered with the same discipline, which this
  Python implementation does not.
- Up to about 10 domains there is no advantage: a fairly trained monolith matches the swarm (auditor: 84.6%
  vs 84.2% at N = 10). The architecture earns its keep when the workload is wide.

Where it fails: when the domain cannot be recognised from the input (populations under noise, edge or
adversarial conditions), the router has no signal and the swarm degrades to voting; and when a specialist
must learn a second task, it forgets the first. Both are properties of this routing signal and this brain,
not of the swarm structure, and both were measured rather than argued away.
