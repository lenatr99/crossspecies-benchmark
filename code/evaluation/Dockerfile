# syntax=docker/dockerfile:1
#
# Reproducible container for marker-label and avgBIO evaluation workflows.
FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="crossspecies-evaluation"
LABEL org.opencontainers.image.description="Reproducible marker-label and avgBIO evaluation workflows"
LABEL org.opencontainers.image.source="code/evaluation"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.lock.txt /tmp/requirements.lock.txt

RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba run -n base python -m pip install --no-cache-dir -r /tmp/requirements.lock.txt \
    && micromamba clean --all --yes \
    && rm -f /tmp/environment.yml /tmp/requirements.lock.txt

ENV PATH="/opt/conda/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/code/evaluation \
    MPLCONFIGDIR=/workspace/data/evaluation/.cache/matplotlib \
    NUMBA_CACHE_DIR=/workspace/data/evaluation/.cache/numba \
    XDG_CACHE_HOME=/workspace/data/evaluation/.cache/xdg \
    LOKY_MAX_CPU_COUNT=1 \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /workspace
