# syntax=docker/dockerfile:1
#
# Reproducible container for the ortholog consensus pipeline.
#
# Notes:
# - The runner builds with --platform linux/amd64 intentionally: Bioconda has
#   much broader x86_64 coverage than linux/arm64 for phylogenomics tools.
#   Docker Desktop on Apple Silicon runs this via emulation.
# - Exact package builds are locked in environment.docker.lock.yml.
FROM mambaorg/micromamba:1.5.10@sha256:e3797091302382ea841498bc93a7b0a50f7c1448333d5e946d2d1608d0c5f43d

LABEL org.opencontainers.image.title="crossspecies-orthologs"
LABEL org.opencontainers.image.description="Reproducible OrthoFinder/OrthoDB/OMA ortholog consensus pipeline"
LABEL org.opencontainers.image.source="code/orthologs"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.docker.lock.yml /tmp/environment.docker.lock.yml

RUN micromamba install -y -n base -f /tmp/environment.docker.lock.yml \
    && micromamba clean --all --yes \
    && rm -f /tmp/environment.docker.lock.yml

ENV PATH="/opt/conda/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MAFFT_BINARIES=

WORKDIR /workspace/code/orthologs
