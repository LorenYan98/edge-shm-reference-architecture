#!/bin/bash
# After the re-scoring run: fold Chillon's masked ViT scores into out/, re-time the PCA
# detector in the container on an idle host. Idempotent.
cd "$(dirname "$0")"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
while [ ! -f out/rescore/DONE ]; do sleep 60; done
while pgrep -f "shm.rescore" > /dev/null; do sleep 10; done
log "rescore done"
python3 -m shm.merge_rescore && log "merge done"
rm -f out/PCA_LATENCY_DONE
docker run --rm --cpus 4 --memory 4g -v "$PWD":/work -w /work shm-edge:py37 \
  python -m shm.latency_pca --threads 4 && touch out/PCA_LATENCY_DONE && log "pca latency re-timed"
touch out/V2_FINISHED; log "all done"
