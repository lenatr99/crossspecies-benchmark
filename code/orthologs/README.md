# Ortholog consensus pipeline

Multi-evidence mapping of query genes (for example *Dictyostelium discoideum* and *Caenorhabditis elegans*) to human orthologs. The recommended run path is **Docker** so conda packages (OrthoFinder, pandas, openpyxl, etc.) match the pinned Linux builds in [`environment.docker.lock.yml`](environment.docker.lock.yml).

## Quick start (Docker)

Install [Docker](https://docs.docker.com/get-docker/). From the **repository root**:

```bash
bash code/orthologs/orthologs_run_all_docker.sh
```

Typical run with 16 CPUs for Docker and OrthoFinder, **reusing an existing image** (no rebuild):

```bash
ORTHOLOGS_DOCKER_CPUS=16 ORTHOFINDER_THREADS=16 ORTHOLOGS_DOCKER_BUILD=0 bash code/orthologs/orthologs_run_all_docker.sh
```

More pipeline overrides (passed through to the Python driver):

```bash
bash code/orthologs/orthologs_run_all_docker.sh --threads 12
bash code/orthologs/orthologs_run_all_docker.sh --skip-oma
```

## Docker image, layout, manifests

The image **`crossspecies-orthologs:2026-04-28`** (override with **`ORTHOLOGS_DOCKER_IMAGE`**) is built from:

- [`Dockerfile`](Dockerfile)
- [`environment.docker.yml`](environment.docker.yml) (intent / channels)
- [`environment.docker.lock.yml`](environment.docker.lock.yml) — exact Linux package builds used inside the container

Pipeline data are **not** baked into the image. The repo is mounted at **`/workspace`**; caches and outputs stay under:

- **`data/`**
- **`results/orthologs/`**

The image defaults to **`linux/amd64`** because Bioconda has the broadest phylogenomics tooling there; Docker Desktop on Apple Silicon runs it via emulation unless you override **`ORTHOLOGS_DOCKER_PLATFORM`**.

Unless **`ORTHOLOGS_DOCKER_BUILD=0`**, **`orthologs_run_all_docker.sh`** runs **`docker build`** first; Docker reuses layers when **`Dockerfile`** and the lock file are unchanged.

Each Docker run writes **`results/orthologs/docker_runtime_manifest.json`** (image tag, image ID, platform, paths to Dockerfile and env specs, processor/thread env).

## Useful environment variables

Highlights (**[`orthologs_run_all_docker.sh`](orthologs_run_all_docker.sh)** documents the rest, including OMA worker parallelism):

- **`ORTHOLOGS_DOCKER_BUILD=0`** — skip **`docker build`** when the image already matches Dockerfile + lock file.
- **`ORTHOLOGS_DOCKER_IMAGE`** — tag to build and run (default **`crossspecies-orthologs:2026-04-28`**).
- **`ORTHOLOGS_DOCKER_PLATFORM`** — typically **`linux/amd64`**.
- **`ORTHOLOGS_DOCKER_CPUS`** — CPU quota for **`docker run`**.
- **`ORTHOFINDER_THREADS`** — OrthoFinder parallelism inside the container.
- **`ORTHOLOGS_OMA_WORKERS`** — optional thread count override for dicty-human OMA xrefs (**01d** only).

After changing **`Dockerfile`**, **`environment.docker.lock.yml`**, or dependencies they reference, omit **`ORTHOLOGS_DOCKER_BUILD=0`** once (or **`docker build`** manually) before relying on **`ORTHOLOGS_DOCKER_BUILD=0`** again.

## Data provenance (static inputs)

Step manifests record SHA-256 and paths for downloads when applicable. Important static inputs include:

**`data/Homo_sapiens.gene_info`**

- Source: [`https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz`](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz) (gzip downloaded and decompressed locally).
- The Docker runner downloads and decompresses this during its static-input preflight when missing.
- Download metadata sidecar: **`data/Homo_sapiens.gene_info.gz.metadata.json`**.
- Example SHA-256 (verify against your checkout): **`18e0a3e22ee3afa602083844aaf56abd38eb103f035071b11118cb8f4d328f66`**.
- Purpose: human NCBI Gene ID → approved gene symbol.
- Checksums propagate into **`01d_oma_orthologs_<organism>_manifest.json`** and **`02_consensus_<organism>_manifest.json`**.

**`data/human_ncbi_annotations.txt`**

- Source: OrthoDB gene xrefs filtered to human (**`9606_1`**) and xref type **`NCBIgid`**.
- Purpose: human OrthoDB gene id → NCBI Gene ID before symbol lookup.
- Manifests: same **`01d`** / **`02`** manifests as above.
- The Docker runner derives this during its static-input preflight when missing.

**`data/odb12v2_gene_xrefs.tab.gz`** (or legacy **`data/odb12v1_gene_xrefs.tab.gz`**)

- Source: **`https://data.orthodb.org/current/download/odb_data_dump/odb12v2_gene_xrefs.tab.gz`** (~4.7 GB).
- Purpose: organism and human xref resolution plus OMA ↔ OrthoDB UniProt bridging in **01d**.
- The pipeline accepts either dump filename under `data/`; prefer **`odb12v2_`** (same revision family as **`odb12v2_*`** bulk OG files).
- The Docker runner auto-downloads this file when it is missing. Set **`ORTHOLOGS_DOCKER_BOOTSTRAP_STATIC=0`** to disable static-input preflight.

**`data/orthodb/odb12v1/bulk/odb12v2_*.tab.gz`**

- Source: **`https://data.orthodb.org/current/download/odb_data_dump/`** — Orthogroup membership and related bulk evidence used by OrthoDB steps.

## Non-Docker orchestration

**[`orthologs_run_all.sh`](orthologs_run_all.sh)** runs the same logical steps with your **host** conda and tools; useful only if OrthoFinder, Python pins, etc. match what you expect on that machine.
