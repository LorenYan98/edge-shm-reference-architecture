# Edge-node emulation image: CPU-only torch, same Python minor as the host.
FROM python:3.7-slim
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==1.13.1+cpu \
 && pip install --no-cache-dir numpy==1.21.6 scipy==1.7.3
WORKDIR /work
