# Historical research provenance

This document describes the original development, not the Git history of this fresh publication. Personal paths and private session references are redacted. Original hash identities remain for provenance only; they are not reachable commits here. See PUBLICATION_PROVENANCE.json for byte-level differences.

# Provenance of the Mini Sentinel Swarm project and the Flat Swarm V1 baseline

This project was not started clean. It is the continuation of work done inside `the private origin repository`, and
this file records where every part came from, who contributed what, and where the unsuccessful attempts are
kept. Nothing below is to be read as a claim of novelty.

## Origin

| item | value |
|---|---|
| original repository | PRIVATE_ORIGIN |
| original directory | `uai32-swarm/` |
| original pull request | PRIVATE_ORIGIN (branch `claude/loving-turing-5z6uex`) |
| frozen development head | `530ec02` (private historical reference) (8 October 2026), tagged `flat-swarm-v1` in the original repository |
| extraction | `git subtree split --prefix=uai32-swarm` of that head; the extracted commits keep authorship, timestamps, messages and file history but have **new SHAs**. `HISTORY.md` maps every original SHA to its extracted SHA. The extracted tree is byte-identical to `uai32-swarm/` at `530ec02`. Extracted SHAs are never to be quoted as the originals. |
| layout commit | the first commit unique to this repository moves the extracted files into `flat-swarm-v1/`, vendors the µAI-32 v1.0.0 binary and source under `uai32/`, and adds `README.md`, `PROVENANCE.md` and `HISTORY.md`. No file under `flat-swarm-v1/` differs from the original. |
| historical tag in private repository | `flat-swarm-v1` on the original layout commit; not imported into this publication |

## Lineage of the brain

Flat Swarm V1 runs the released **µAI-32 v1.0.0** binary unmodified: release tag
`uai32-v1.0.0` (private historical reference) at commit `e651c1c`,
executable 9,188 bytes, SHA-256 `996d73cba9676e8932206b9c80493becde6d04b6a57b70839999b0824c0186cb`.
µAI-32 itself was built for the Sentovara µAI-32 challenge (pull request
#2 (private historical reference), merged), was found to have assurance defects by an
independent audit, was fixed in one corrective commit by Jonathan (`c2c1497`) and one by Claude (`3a420c8`),
received a final corrective commit for two further audit findings (`3e924e6`), and was passed by the final
independent audit on 7 October 2026 (28 of 28 verification checks, 18 regression groups, 222 subprocess cases,
9,781 of 10,000 MNIST test digits, 9,188 + 21,468 = 30,656 bytes). The release review landed as pull request
#3 (private historical reference). Other material from the original collection is outside this publication.

## Who contributed what

| contributor | contribution |
|---|---|
| **Jonathan (GitHub `jonny5isalive5`) / Sentovara** | the original swarm concept and the cog-network concept that follows it; architecture direction; the constraints (one unmodified tiny brain, no larger model at run time, total resource cost must keep the brain's advantage); the challenge definition with its stages, verification rule (build → test → audit → one remediation → retest) and finding categories; the decision to freeze Flat Swarm V1 as the control; owner of the original repository and author of the µAI-32 corrective commit `c2c1497` |
| **Fable 5.1 / Claude** (Anthropic, model `claude-fable-5-1`, Claude Code cloud session, 6 to 8 October 2026; session PRIVATE_SESSION_REDACTED) | primary implementation of µAI-32 and of the swarm: `swarm.py`, the synthetic task world, library, router, arbitration, composition, measurements, all stages; the experiments and their reruns; debugging (the memory flaw at N = 1,000, the stale-file reproducibility bug, the measurement artefacts); the single remediation after the audit; `FINAL.md`, `FAILURES.md`, this split and these provenance documents |
| **ChatGPT (OpenAI), working with Jonathan as "Codex" in the notes** | experimental framing and architecture discussion; the evidence and audit discipline reflected in the challenge definition; review contributions on the preceding work where applicable, including audit specifications for µAI-32; coordination of the repository split |
| **Independent auditor of µAI-32** (engaged by Jonathan; model, version and tooling not recorded in the repositories) | the audits of `82bd1f0`, `c2c1497` and `3e924e6` that found the assurance defects and finally passed the release (7 October 2026) |
| **Independent auditor of Flat Swarm V1** (a separate Claude Fable 5.1 agent instance, model `claude-fable-5-1`, spawned by the implementing session with a read-only copy of the directory, 8 October 2026) | the audit pass: zero BLOCKER, eleven IMPORTANT, six LATER findings (under-trained monolith baselines, metadata excluded from "equal storage", swapped timing labels, understated working set and routing cost, two hard-coded measurements, an oracle composition table, routing trivially separable by construction, shared library names, reproducibility drift in the swarm, naming); the retest after remediation (14 fixed, 3 partially fixed with cosmetic residue, no open IMPORTANT item). Its independence is that of a separate agent context with no write access, not of a different organisation or a human; readers should weigh it accordingly. |
| **Future contributors** | to be credited here, by name, for the specific thing they contribute (an experiment, a fix, an audit, a dataset), with the commit or pull request that carries it |

## Where the failures are

Every unsuccessful attempt is kept: `flat-swarm-v1/FAILURES.md` (items 1 to 14: two wrong RSS measurements,
a router that always chose the broad "noisy" expert, the 7.5 GiB unchunked router at N = 1,000, baselines
trained further by stale files, total forgetting, crude spawning, the under-trained baselines found by the
audit, the audit and retest records), the superseded result files in `flat-swarm-v1/results/`
(`stage6_superseded_old_router.*`, `stage6_superseded_unchunked_router.*`), and the original commit messages
in `HISTORY.md`, which say what was wrong at each step.

## What Flat Swarm V1 is, exactly

The determination in `flat-swarm-v1/FINAL.md` is the whole claim. Its limitations are part of the baseline and
are not to be broadened: domain identity is explicit in the synthetic task (routing is a trivially separable
nearest-centroid problem); routing cannot tell same-domain experts apart under noise, edge or adversarial
conditions; forgetting inside a specialist is total; spawning clusters crudely; the Python orchestrator costs
39 to 489 MB and routing grows to 112 µs per row at 1,000 specialists; the equal-storage comparison at
N = 1,000 is capped. Flat Swarm V1 is the control against which Cog Network V1 will be measured, and the
control does not move.
