#!/bin/sh
# run_all.sh -- reproduce every stage of the Mini Sentinel swarm experiment from scratch.
# Needs: the released uai32 binary at ../uai32/dist/uai32 (or UAI32=...), python3 + numpy, a C compiler (for the
# RSS launcher).  Everything is seeded (--seed 1); results/*.json and results/REPORT.md are rewritten.
set -eu
cd "$(dirname "$0")"
SEED=${SEED:-1}
rm -rf work results/stage[1-8].json results/stretch.json results/stage2_provenance.jsonl   # start clean: uai32 train continues an existing model file
python3 swarm.py 1 2 3 4 5 7 8 stretch --seed "$SEED"
python3 swarm.py 6 --seed "$SEED" --sizes "${SIZES:-2,10,100,1000}"
python3 report.py
