"""Internal helpers shared by the multi-evidence ortholog pipeline.

Kept under a leading-underscore name to make it clear this is not a step in
the numbered pipeline and is not invoked directly by ``orthologs_run_all.sh``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import email.utils
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=sort_keys)
        handle.write("\n")


def open_maybe_gzip(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return path.open(mode, encoding=encoding)


# ---------------------------------------------------------------------------
# HTTP fetch with cache + retry
# ---------------------------------------------------------------------------


def fetch_text(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 120,
    sleep_seconds: float = 0.2,
    max_retries: int = 3,
    backoff_seconds: tuple[float, ...] = (1.0, 4.0, 16.0),
    accept_404: bool = False,
) -> tuple[int, str]:
    """GET ``url`` with exponential backoff on 429/5xx.

    Returns ``(status_code, body_text)``. ``body_text`` is empty on 404 when
    ``accept_404`` is set so callers can persist a cached "no result" record.
    """

    if sleep_seconds:
        time.sleep(sleep_seconds)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params or {}, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
            continue

        if response.status_code == 404 and accept_404:
            return 404, ""
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            elif retry_after:
                try:
                    parsed = email.utils.parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    parsed = None
                delay = (
                    max(0.0, parsed.timestamp() - time.time())
                    if parsed
                    else backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                )
            else:
                delay = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.status_code, response.text

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


def stream_download_to_file(
    url: str,
    dest: Path,
    *,
    label: str = "",
    chunk_size: int = 4 * 1024 * 1024,
    sleep_seconds: float = 0.2,
    connect_timeout: int = 120,
    read_timeout: int = 600,
) -> tuple[int, str]:
    """Stream ``url`` to ``dest`` without loading the whole body into RAM.

    Shows a tqdm byte counter (GiB/MiB) while hashing incrementally.

    Returns ``(byte_count, sha256_hex)``.
    """

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "Binary downloads require tqdm. Install with: pip install tqdm "
            "(see code/orthologs/requirements.txt)"
        ) from exc

    if sleep_seconds:
        time.sleep(sleep_seconds)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()

    digest = hashlib.sha256()
    written = 0

    print(f"Downloading → {dest.name}", flush=True)
    print(url, flush=True)

    response = requests.get(
        url,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )
    try:
        response.raise_for_status()
        cl = response.headers.get("Content-Length")
        total_b: int | None = int(cl) if cl and cl.isdigit() else None

        desc = label if label else dest.name

        with part.open("wb") as handle, tqdm(
            total=total_b,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            miniters=1,
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                pbar.update(len(chunk))

        part.replace(dest)
    except Exception:
        if part.exists():
            part.unlink(missing_ok=True)
        raise
    finally:
        response.close()

    return written, digest.hexdigest()


def cached_download(
    url: str,
    cache_path: Path,
    *,
    refresh: bool = False,
    binary: bool = False,
    metadata_extra: dict[str, Any] | None = None,
    download_label: str = "",
) -> Path:
    """Download ``url`` into ``cache_path`` unless cached, write sidecar manifest.

    Binary downloads are streamed to disk (multi‑GB OrthoDB tables never fully
    buffer in RAM). ``download_label`` sets the tqdm progress ``desc``.

    Returns ``cache_path``. Sidecar metadata path is ``cache_path + ".metadata.json"``.
    """

    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    if cache_path.exists() and not refresh:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        _n, digest = stream_download_to_file(url, cache_path, label=download_label)
    else:
        _, payload = fetch_text(url, sleep_seconds=0.5)
        cache_path.write_text(payload, encoding="utf-8")
        digest = sha256_text(payload)

    metadata: dict[str, Any] = {
        "url": url,
        "retrieved_utc": utc_now(),
        "sha256": digest,
        "bytes": cache_path.stat().st_size,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    write_json(metadata_path, metadata)
    return cache_path


# ---------------------------------------------------------------------------
# OrthoDB xrefs streaming
# ---------------------------------------------------------------------------


def stream_xrefs(
    xrefs_path: Path,
    *,
    species_prefixes: Iterable[str] | None = None,
    xref_types: Iterable[str] | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Stream OrthoDB ``*_gene_xrefs.tab.gz`` filtered by species and xref type.

    On disk this is typically ``data/odb12v2_gene_xrefs.tab.gz`` (OrthoDB dump) or the
    legacy name ``odb12v1_gene_xrefs.tab.gz``.

    The full table is ~5 GB compressed; callers MUST scope the stream with
    ``species_prefixes`` (matched against the start of the OrthoDB gene id)
    or ``xref_types`` to keep memory + runtime manageable.

    For interactive shell slicing of this gzip, parallel decompression
    (`pigz -dc file.gz`) is typically faster than `gzcat`, though the
    bottleneck is often awk rather than inflate.
    """

    species_set = set(species_prefixes) if species_prefixes else None
    type_set = set(xref_types) if xref_types else None
    with gzip.open(xrefs_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            orthodb_id, external_id, external_type = parts
            if type_set is not None and external_type not in type_set:
                continue
            if species_set is not None:
                if not any(orthodb_id.startswith(prefix) for prefix in species_set):
                    continue
            yield orthodb_id, external_id, external_type


def load_xrefs_subset(
    xrefs_path: Path,
    *,
    species_prefixes: Iterable[str],
    xref_types: Iterable[str],
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[tuple[str, str], str]:
    """Load a small subset of OrthoDB xrefs into memory, with optional disk cache.

    Returns ``{(orthodb_id, xref_type): external_id}``. When ``cache_path`` is
    provided, the streamed subset is materialised to a small TSV so future runs
    skip the multi-minute scan of the 5 GB master file.
    """

    species_set = sorted(set(species_prefixes))
    type_set = sorted(set(xref_types))

    if cache_path and cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
        result: dict[tuple[str, str], str] = {}
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                orthodb_id, xref_type, external_id = line.rstrip("\n").split("\t")
                result[(orthodb_id, xref_type)] = external_id
        return result

    result = {}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") if cache_path else _null_writer() as out:
        for orthodb_id, external_id, external_type in stream_xrefs(
            xrefs_path,
            species_prefixes=species_set,
            xref_types=type_set,
        ):
            result[(orthodb_id, external_type)] = external_id
            if cache_path:
                out.write(f"{orthodb_id}\t{external_type}\t{external_id}\n")
    return result


@contextmanager
def _null_writer() -> Iterator[io.TextIOBase]:
    yield io.StringIO()


# ---------------------------------------------------------------------------
# OG id parsing
# ---------------------------------------------------------------------------


def og_level_taxid(og_id: str) -> str | None:
    """Return the level taxid encoded in OrthoDB OG ids of the form ``<n>at<taxid>``."""

    if "at" not in og_id:
        return None
    _, _, tail = og_id.partition("at")
    return tail or None


# ---------------------------------------------------------------------------
# FASTA parsing (one canonical OrthoDB protein per record)
# ---------------------------------------------------------------------------


def iter_orthodb_fasta(
    fasta_path: Path,
) -> Iterator[tuple[str, dict[str, Any], str]]:
    """Yield ``(orthodb_id, header_metadata, sequence)`` per record.

    Headers look like ``>352472_0:000000 {"organism": ..., "pub_gene_id": ...}``.
    """

    current_id: str | None = None
    current_meta: dict[str, Any] = {}
    seq_parts: list[str] = []
    with fasta_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, current_meta, "".join(seq_parts)
                header = line[1:]
                first_token, _, rest = header.partition(" ")
                current_id = first_token
                current_meta = {}
                rest = rest.strip()
                if rest.startswith("{"):
                    try:
                        current_meta = json.loads(rest)
                    except json.JSONDecodeError:
                        current_meta = {"raw": rest}
                else:
                    current_meta = {"raw": rest}
                seq_parts = []
            else:
                seq_parts.append(line)
    if current_id is not None:
        yield current_id, current_meta, "".join(seq_parts)


def write_renamed_fasta(
    src_fasta: Path,
    dst_fasta: Path,
    *,
    species_tag: str,
) -> tuple[int, int]:
    """Rewrite an OrthoDB FASTA with deterministic ``>{species}|{orthodb_id}`` headers.

    Drops duplicate orthodb_ids (defensive — OrthoDB FASTA should have one
    canonical protein per record). Returns ``(records_in, records_out)``.
    """

    seen: set[str] = set()
    n_in = 0
    n_out = 0
    dst_fasta.parent.mkdir(parents=True, exist_ok=True)
    with dst_fasta.open("w", encoding="utf-8") as out:
        for orthodb_id, _meta, seq in iter_orthodb_fasta(src_fasta):
            n_in += 1
            if orthodb_id in seen:
                continue
            seen.add(orthodb_id)
            out.write(f">{species_tag}|{orthodb_id}\n")
            for chunk in _wrap(seq, 60):
                out.write(chunk + "\n")
            n_out += 1
    return n_in, n_out


def _wrap(text: str, width: int) -> Iterator[str]:
    for i in range(0, len(text), width):
        yield text[i : i + width]


# ---------------------------------------------------------------------------
# Human symbol resolution
# ---------------------------------------------------------------------------


def load_human_symbols(
    human_xrefs_path: Path,
    gene_info_path: Path,
) -> dict[str, str]:
    """Resolve human OrthoDB ids → HGNC gene symbol via NCBI gene_info."""

    human_to_ncbi: dict[str, str] = {}
    with human_xrefs_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2] == "NCBIgid":
                human_to_ncbi[parts[0]] = parts[1]

    ncbi_to_symbol: dict[str, str] = {}
    with gene_info_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                ncbi_to_symbol[parts[1]] = parts[2]

    return {
        orthodb_id: ncbi_to_symbol[ncbi_id]
        for orthodb_id, ncbi_id in human_to_ncbi.items()
        if ncbi_id in ncbi_to_symbol
    }


def ensure_human_gene_info(gene_info_path: Path, gene_info_url: str) -> Path:
    """Download and decompress NCBI human ``gene_info`` when absent."""

    if gene_info_path.exists() and gene_info_path.stat().st_size > 0:
        return gene_info_path

    gzip_path = Path(str(gene_info_path) + ".gz")
    cached_download(gene_info_url, gzip_path, binary=True, download_label="human gene_info")

    part = gene_info_path.with_suffix(gene_info_path.suffix + ".part")
    print(f"Decompressing {gzip_path.name} -> {gene_info_path.name}")
    gene_info_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gzip_path, "rb") as src, part.open("wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
    part.replace(gene_info_path)
    return gene_info_path


def ensure_human_ncbi_annotations(
    human_xrefs_path: Path,
    orthodb_xrefs_path: Path,
) -> Path:
    """Materialize human OrthoDB id -> NCBI GeneID rows from the full xref dump."""

    if human_xrefs_path.exists() and human_xrefs_path.stat().st_size > 0:
        return human_xrefs_path
    if not orthodb_xrefs_path.exists():
        raise FileNotFoundError(
            f"Cannot create {human_xrefs_path}: missing {orthodb_xrefs_path}"
        )

    part = human_xrefs_path.with_suffix(human_xrefs_path.suffix + ".part")
    count = 0
    print(f"Creating {human_xrefs_path} from {orthodb_xrefs_path.name}")
    human_xrefs_path.parent.mkdir(parents=True, exist_ok=True)
    with part.open("w", encoding="utf-8") as out:
        for orthodb_id, external_id, external_type in stream_xrefs(
            orthodb_xrefs_path,
            species_prefixes=["9606_1"],
            xref_types=["NCBIgid"],
        ):
            out.write(f"{orthodb_id}\t{external_id}\t{external_type}\n")
            count += 1
    part.replace(human_xrefs_path)
    print(f"Human NCBI annotation rows: {count}")
    return human_xrefs_path


def load_human_uniprot_to_orthodb(
    xrefs_path: Path,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, str]:
    """Build ``{uniprot_acc: human_orthodb_id}`` from the OrthoDB xrefs table."""

    if cache_path and cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
        return {
            row.split("\t", 1)[0]: row.rstrip("\n").split("\t", 1)[1]
            for row in cache_path.read_text(encoding="utf-8").splitlines()
            if "\t" in row
        }

    mapping: dict[str, str] = {}
    for orthodb_id, external_id, external_type in stream_xrefs(
        xrefs_path,
        species_prefixes=["9606_1"],
        xref_types=["UniProt"],
    ):
        mapping[external_id] = orthodb_id

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as out:
            for uniprot_acc, orthodb_id in sorted(mapping.items()):
                out.write(f"{uniprot_acc}\t{orthodb_id}\n")
    return mapping
