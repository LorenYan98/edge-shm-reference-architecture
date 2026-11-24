#!/bin/bash
# PCA detector latency: fit on host, time in the 4 vCPU / 4 GB container.
cd "$(dirname "$0")"
python3 -m shm.latency_pca --fit && \
docker run --rm --cpus 4 --memory 4g -v "$PWD":/work -w /work shm-edge:py37 \
  python -m shm.latency_pca --threads 4 && touch out/PCA_LATENCY_DONE
