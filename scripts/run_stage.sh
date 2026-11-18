#!/usr/bin/env bash
# usage: run_stage.sh STEPS OUTDIR "SEEDS" "ROLES"
cd "$(dirname "$0")/.."
STEPS=$1; OUT=$2; SEEDS=$3; ROLES=$4
for s in $SEEDS; do for r in $ROLES; do
  [ -f $OUT/${r}_s${s}.json ] && continue
  python3 -m src.train --role $r --seed $s --steps $STEPS --chunk 10000 --out $OUT > logs/train_${r}_s${s}.log 2>&1
done; done
echo DONE > $OUT/done.flag
