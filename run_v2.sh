#!/bin/bash
# v2 pipeline: per-structure front ends (shm/frontend.py) and one MAE-ViT per structure.
# Idempotent and resumable: re-running skips finished caches/models (train.py resumes from last.pt).
#   cache  : python3 -m shm.datasets <S> <limit> v2      (already built by hand, re-built here if missing)
#   train  : three trainers in parallel, 10 threads each, >= 1500 optimizer steps each
#   after  : evaluate -> raw sample -> container latency -> out/PIPELINE_DONE
cd "$(dirname "$0")"
set -u
STRUCTS="KW51 VANERSBORG CHILLON"
RUNS="KW51=v2_KW51,VANERSBORG=v2_VANERSBORG,CHILLON=v2_CHILLON"
HW="development host (32-core CPU); edge latency measured in a 4-vCPU, 4 GB container"
mkdir -p logs_v2
log() { echo "[$(date -u +%FT%TZ)] $*"; }

declare -A LIMIT=( [KW51]=0 [VANERSBORG]=0 [CHILLON]=44 )
for S in $STRUCTS; do
  while pgrep -f "shm.datasets $S" > /dev/null; do sleep 30; done      # a hand-started cache build
  [ -f cache/v2/$S.npz ] || python3 -m shm.datasets $S ${LIMIT[$S]} v2 > logs_v2/cache_$S.log 2>&1
done
log "caches ready: $(for S in $STRUCTS; do echo -n "$S=$(($(wc -l < cache/v2/$S.csv)-1)) "; done)"

for S in $STRUCTS; do
  if [ ! -f runs/v2_$S/model.pt ]; then
    python3 -m shm.train --run v2_$S --frontend v2 --structures $S --epochs 15 --min-steps 1500 \
      --per-structure 2500 --threads 10 >> logs_v2/train_$S.log 2>&1 &
    log "training v2_$S started (pid $!)"
  fi
done
wait
for S in $STRUCTS; do [ -f runs/v2_$S/model.pt ] || { log "v2_$S has no model.pt, aborting"; exit 1; }; done
log "all models ready"

python3 -m shm.evaluate --runs $RUNS --frontend v2 --threads 30 --hardware "$HW" && log "evaluate done"
python3 -m shm.raw_sample --frontend v2 --per-structure 400 && log "raw sample done"
docker run --rm --cpus 4 --memory 4g -v "$PWD":/work -w /work shm-edge:py37 \
  python -m shm.latency --runs $RUNS --frontend v2 --threads 4 && log "latency done"
touch out/PIPELINE_DONE; log "all done"
