"""Step 01d: OMA REST API orthologs (third independent evidence source).

For most organisms each gene maps to UniProt via OrthoDB xrefs, then we call
``GET …/protein/{id}/orthologs/``. Remote species lists sometimes omit mammals;
*D. discoideum* also has almost no OrthoDB UniProt xrefs.

For dicty we derive gene ids from OrthoDB FASTA ``pub_gene_id``, then consume
paginated pairwise ``GET …/pairs/DICDI/9606/`` mappings (the same relationships
shown in OMA genome–genome matrices), resolving human ``canonicalid`` symbols
via per-protein ``…/xref/`` stubs to UniProt accessions compatible with OrthoDB.
Each fetched JSON is cached locally.

Pairwise ``rel_type`` semantics follow ``1:1``, ``1:m``, ``m:n``, etc.; we normalize
lightly for downstream consensus tiers.

Human symbols use the OrthoDB UniProt xref table for Homo sapiens plus ``gene_info``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from threading import Lock
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from tqdm.auto import tqdm

from _utils import (
    ensure_human_gene_info,
    ensure_human_ncbi_annotations,
    fetch_text,
    iter_orthodb_fasta,
    load_human_symbols,
    load_human_uniprot_to_orthodb,
    load_xrefs_subset,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
)
from config import (
    HUMAN_GENE_INFO,
    HUMAN_GENE_INFO_URL,
    HUMAN_NCBI_ANNOTATIONS,
    HUMAN_NCBI_ANNOTATIONS_NOTE,
    HUMAN_TAXID,
    OMA_API_BASE,
    OMA_CACHE_DIR,
    ORGANISMS,
    ORTHODB_BULK_BASE,
    ORTHODB_CACHE_DIR,
    ORTHODB_GENE_XREFS,
    ORTHODB_GENE_XREFS_DUMP,
    ORTHODB_GENE_XREFS_LEGACY,
    RESULTS_DIR,
)


XREF_CACHE_DIR = ORTHODB_CACHE_DIR / "xrefs"


# Thread-safe on-disk cache writes when using ThreadPoolExecutor for xref joins.
_FETCH_PATH_LOCKS: dict[str, threading.Lock] = {}
_FETCH_PATH_MASTER = threading.Lock()


def _path_lock_for(cache_path: Path) -> threading.Lock:
    key = str(cache_path.resolve())
    with _FETCH_PATH_MASTER:
        if key not in _FETCH_PATH_LOCKS:
            _FETCH_PATH_LOCKS[key] = threading.Lock()
        return _FETCH_PATH_LOCKS[key]


_HUMAN_SYM_CACHE_GUARD = threading.Lock()


def map_to_uniprot(
    organism: str,
    organism_orthodb_prefix: str,
    *,
    xrefs_path: Path,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """Return ``{organism_orthodb_id: [uniprot_acc, ...]}`` from OrthoDB xrefs."""

    cache_path = XREF_CACHE_DIR / f"{organism}_uniprot_xrefs.tsv"
    pairs = load_xrefs_subset(
        xrefs_path,
        species_prefixes=[organism_orthodb_prefix],
        xref_types=["UniProt"],
        cache_path=cache_path,
        refresh=refresh,
    )
    out: dict[str, list[str]] = defaultdict(list)
    for (orthodb_id, _xref_type), external_id in pairs.items():
        out[orthodb_id].append(external_id)
    return out


def map_to_pub_gene_ids(fasta_path: Path) -> dict[str, list[str]]:
    """Return ``{orthodb_id: [pub_gene_id, ...]}`` from cached OrthoDB FASTA headers."""

    out: dict[str, list[str]] = defaultdict(list)
    for orthodb_id, meta, _seq in iter_orthodb_fasta(fasta_path):
        if not isinstance(meta, dict):
            continue
        pub = meta.get("pub_gene_id")
        if pub:
            out[orthodb_id].append(str(pub).strip())
    return out


def fetch_oma_orthologs(
    protein_query_id: str,
    *,
    cache_dir: Path,
    refresh: bool = False,
    error_log: Path,
    request_sleep_seconds: float = 0.2,
    error_log_lock: Lock | None = None,
) -> dict[str, object] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{protein_query_id}.json"
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")

    if cache_path.exists() and not refresh:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    lk = _path_lock_for(cache_path)
    with lk:
        if cache_path.exists() and not refresh:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        url = f"{OMA_API_BASE}/protein/{protein_query_id}/orthologs/"
        try:
            status, text = fetch_text(
                url, sleep_seconds=request_sleep_seconds, accept_404=True
            )
        except Exception as exc:  # noqa: BLE001
            error_log.parent.mkdir(parents=True, exist_ok=True)
            log_line = (
                json.dumps(
                    {
                        "protein_query_id": protein_query_id,
                        "url": url,
                        "error": str(exc),
                        "ts": utc_now(),
                    }
                )
                + "\n"
            )
            if error_log_lock is not None:
                with error_log_lock:
                    with error_log.open("a", encoding="utf-8") as handle:
                        handle.write(log_line)
            else:
                with error_log.open("a", encoding="utf-8") as handle:
                    handle.write(log_line)
            return None

        body: object
        if status == 404:
            body = {"orthologs": [], "_status": 404, "_query": protein_query_id}
        else:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {"raw": text, "_status": status, "_query": protein_query_id}

        cache_path.write_text(
            json.dumps(body, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_json(
            metadata_path,
            {
                "url": url,
                "status": status,
                "protein_query_id": protein_query_id,
                "retrieved_utc": utc_now(),
                "sha256": sha256_text(json.dumps(body, sort_keys=True)),
            },
        )
        return body if isinstance(body, (list, dict)) else None


def _normalise_rel_type(value: str | None) -> str:
    if not value:
        return "n/a"
    v = value.replace(":", "").lower().strip()
    if v in ("11", "1to1", "11ortholog", "oneone"):
        return "1:1"
    if v in ("1n", "1m"):
        return "1:n"
    if v in ("n1", "m1"):
        return "n:1"
    if v in ("nm", "mn", "manymany"):
        return "n:m"
    return value


def _extract_human_orthologs(
    response: object,
) -> list[tuple[str, str, str, int | str | None]]:
    """Return ``[(canonical_id, rel_type, omaid, entry_nr), ...]`` for human orthologs.

    OMA's REST schema occasionally varies between releases; we accept the
    common shapes (top-level list, ``{"orthologs": [...]}``, paginated dict).
    ``canonicalid`` may be a UniProt accession or an OMA/SwissProt mnemonic
    such as ``NDRG4_HUMAN``; callers resolve it to UniProt when needed.
    """

    if response is None:
        return []
    if isinstance(response, dict):
        items = response.get("orthologs") or response.get("data") or response.get("results") or []
        if not items and "rel_type" in response:
            items = [response]
    elif isinstance(response, list):
        items = response
    else:
        return []

    out: list[tuple[str, str, str, int | str | None]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        species_taxid = (
            (item.get("species") or {}).get("taxon_id")
            if isinstance(item.get("species"), dict)
            else item.get("taxon_id")
        )
        if str(species_taxid) != HUMAN_TAXID:
            continue
        rel_type = _normalise_rel_type(
            item.get("rel_type") or item.get("relType") or item.get("relation")
        )
        canonical_id = item.get("canonicalid") or item.get("canonical_id") or ""
        omaid = item.get("omaid") or ""
        if canonical_id:
            out.append((str(canonical_id), rel_type, str(omaid), item.get("entry_nr")))
    return out


def _xref_uniprot_process_one_gene(
    idx: int,
    organism_gene: str,
    *,
    probes_by_gene: dict[str, list[str]],
    cache_dir: Path,
    refresh: bool,
    error_log: Path,
    human_uniprot_to_orthodb: dict[str, str],
    human_symbols: dict[str, str],
    request_sleep_seconds: float,
    error_log_lock: Lock | None,
) -> tuple[int, list[dict[str, str]], int, bool]:
    """Return ``(sort_index, rows, n_oma_queries, skipped_no_probe)`` for one gene."""

    probe_ids = probes_by_gene.get(organism_gene, [])
    if not probe_ids:
        return (idx, [], 0, True)

    rows: list[dict[str, str]] = []
    queried = 0
    for probe in probe_ids:
        response = fetch_oma_orthologs(
            probe,
            cache_dir=cache_dir,
            refresh=refresh,
            error_log=error_log,
            request_sleep_seconds=request_sleep_seconds,
            error_log_lock=error_log_lock,
        )
        queried += 1
        for human_canonical, rel_type, omaid, entry_nr in _extract_human_orthologs(response):
            human_uniprot = human_canonical_to_uniprot_acc(
                human_canonical,
                cache_dir=cache_dir,
                refresh=refresh,
                omaid_fallback=omaid or None,
                entry_nr_fallback=entry_nr,
                request_sleep_seconds=request_sleep_seconds,
            )
            human_orthodb_id = human_uniprot_to_orthodb.get(human_uniprot, "")
            human_symbol = human_symbols.get(human_orthodb_id, "")
            rows.append(
                {
                    "organism_orthodb_id": organism_gene,
                    "organism_query_id": probe,
                    "human_uniprot": human_uniprot,
                    "human_oma_canonicalid": human_canonical,
                    "human_orthodb_id": human_orthodb_id,
                    "human_symbol": human_symbol,
                    "oma_rel_type": rel_type,
                    "oma_id": omaid,
                }
            )
    return (idx, rows, queried, False)


DDB_G_PATTERN = re.compile(r"^DDB_G\d+$")
_PRIMARY_UNIPROT_ACC = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
_A_TREMBL_PREFIX = re.compile(r"^A0A[A-Z0-9]+$")  # UniProt A0A* accessions


def cast_dict_rows(items: list[object]) -> list[dict[str, object]]:
    return [x for x in items if isinstance(x, dict)]


def invert_pub_gene_to_orthodb(fasta_path: Path) -> dict[str, str]:
    """Map ``DDB_G*`` → OrthoDB id (first wins if ambiguous)."""

    out: dict[str, str] = {}
    mapping = map_to_pub_gene_ids(fasta_path)
    for oid, pubs in mapping.items():
        for p in pubs:
            if DDB_G_PATTERN.match(p) and p not in out:
                out[p] = oid
    return out


def _safe_cache_name_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:200]


def _fetch_oma_xref_resilient(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    accept_404: bool = True,
    max_attempts: int = 6,
    request_sleep_seconds: float = 0.2,
) -> list[object]:
    """Fetch ``…/xref/`` with extra backoff on 502/503/504; return ``[]`` if OMA stays unhealthy."""

    delay_s = 1.0
    for attempt in range(max_attempts):
        try:
            body = _fetch_oma_cached_array(
                url,
                cache_path,
                refresh=refresh,
                accept_404=accept_404,
                request_sleep_seconds=request_sleep_seconds,
            )
            return body if isinstance(body, list) else []
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code >= 500 and attempt + 1 < max_attempts:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2.0, 32.0)
                continue
            if code >= 500:
                return []
            raise
        except requests.RequestException:
            if attempt + 1 < max_attempts:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2.0, 32.0)
                continue
            return []
    return []


def _fetch_oma_cached_array(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    accept_404: bool = False,
    request_sleep_seconds: float = 0.2,
    max_retries: int = 8,
    backoff_seconds: tuple[float, ...] = (2.0, 8.0, 20.0, 45.0, 90.0, 180.0),
) -> object | None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    if cache_path.exists() and not refresh:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    lk = _path_lock_for(cache_path)
    with lk:
        if cache_path.exists() and not refresh:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        status, text = fetch_text(
            url,
            sleep_seconds=request_sleep_seconds,
            accept_404=accept_404,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        if status == 404 and accept_404:
            payload: object = []
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"raw": text, "_status": status}
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        write_json(metadata_path, {
            "url": url,
            "status": status,
            "retrieved_utc": utc_now(),
            "sha256": sha256_text(json.dumps(payload, sort_keys=True)),
        })
        return payload


def fetch_all_pairs(
    genome1: str,
    genome2: str,
    *,
    cache_dir: Path,
    refresh: bool,
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Paged ``pairs/{g1}/{g2}/``, cached as one merged JSON list."""

    merged = cache_dir / f"pairs_{_safe_cache_name_component(genome1)}_{genome2}_all.json"
    if merged.exists() and merged.stat().st_size > 0 and not refresh:
        body = json.loads(merged.read_text(encoding="utf-8"))
        n_row = len(body) if isinstance(body, list) else 0
        if show_progress:
            print(f"[pairs {genome1}/{genome2}] merged cache ({n_row} pairwise rows)")
        return body if isinstance(body, list) else []

    aggregate: list[dict[str, object]] = []
    page = 1
    per_page = 100
    bar = tqdm(
        desc=f"pairs {genome1}/{genome2}",
        unit="page",
        disable=not show_progress,
    )
    try:
        while True:
            url = (
                f"{OMA_API_BASE}/pairs/{genome1}/{genome2}/"
                f"?per_page={per_page}&page={page}"
            )
            part_path = (
                cache_dir
                / "pairs_pages"
                / f"{_safe_cache_name_component(genome1)}_{genome2}_p{page:05d}.json"
            )
            chunk = _fetch_oma_cached_array(
                url,
                part_path,
                refresh=refresh,
                request_sleep_seconds=0.5,
                max_retries=12,
                backoff_seconds=(5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
            )
            if not isinstance(chunk, list):
                break
            if not chunk:
                break
            aggregate.extend(cast_dict_rows(chunk))
            bar.update(1)
            bar.set_postfix(rows=len(aggregate))
            if len(chunk) < per_page:
                break
            page += 1
    finally:
        bar.close()

    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate


def omaid_to_pub_gene(
    omaid: str,
    *,
    cache_dir: Path,
    refresh: bool,
    request_sleep_seconds: float = 0.2,
) -> str:
    """Resolve a dicty OMA identifier to ``DDB_G*`` via ``…/xref/``."""

    path = (
        cache_dir
        / "dicty_xref_by_omaid"
        / f"{_safe_cache_name_component(omaid)}.json"
    )
    url = f"{OMA_API_BASE}/protein/{omaid}/xref/"
    body = _fetch_oma_xref_resilient(
        url,
        path,
        refresh=refresh,
        accept_404=True,
        request_sleep_seconds=request_sleep_seconds,
    )
    rows = body if isinstance(body, list) else []
    preferred = ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        lab = (
            row.get("source")
            or row.get("Source")
            or ""
        )
        xref = str(row.get("xref") or "").strip()
        if DDB_G_PATTERN.match(xref):
            if (
                preferred == ""
                and lab in {"SourceAC", "ORF Name"}
            ):
                preferred = xref
            if lab == "SourceAC":
                return xref
    if preferred:
        return preferred
    for row in rows:
        if isinstance(row, dict) and DDB_G_PATTERN.match(str(row.get("xref", "")).strip()):
            return str(row.get("xref", "")).strip()
    return ""


_human_u_cache: dict[str, str] = {}


def _human_uniprot_from_xref_rows(rows: list[object]) -> str:
    """Prefer SwissProt UniProt KB accessions parsed from ``…/xref/`` payloads."""

    candidates: list[tuple[int, int, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = str(row.get("source") or "")
        xref = str(row.get("xref") or "").strip()
        if not xref or not src.startswith("UniProtKB"):
            continue
        if not (
            (_PRIMARY_UNIPROT_ACC.fullmatch(xref))
            or (_A_TREMBL_PREFIX.fullmatch(xref) if xref else False)
        ):
            continue
        pri = 0 if "SwissProt" in src else 1 if "TrEMBL" in src else 2
        seq_rank = {"exact": 0, "modified": 1}.get(str(row.get("seq_match")), 2)
        candidates.append((pri, seq_rank, xref))
    candidates.sort(key=lambda x: (x[0], x[1]))
    if candidates:
        return candidates[0][2]
    for row in rows:
        if not isinstance(row, dict):
            continue
        xref = str(row.get("xref") or "").strip()
        src = str(row.get("source") or "")
        if not xref or not src.startswith("UniProtKB"):
            continue
        if len(xref) != 6 or not xref.isupper():
            continue
        if xref[0] in "OPQ" and xref[1].isdigit():
            return xref
    return ""


def _fetch_human_xrefs_for_probe(
    probe_label: str,
    cache_kind: str,
    *,
    cache_dir: Path,
    refresh: bool,
    request_sleep_seconds: float = 0.2,
) -> list[dict[str, object]]:
    """Cached ``GET /protein/{id}/xref/`` (404-safe)."""

    path = (
        cache_dir
        / "human_xref_probe"
        / f"{cache_kind}_{_safe_cache_name_component(probe_label)}.json"
    )
    url = f"{OMA_API_BASE}/protein/{quote(str(probe_label))}/xref/"
    body = _fetch_oma_xref_resilient(
        url,
        path,
        refresh=refresh,
        accept_404=True,
        request_sleep_seconds=request_sleep_seconds,
    )
    rows = body if isinstance(body, list) else []
    return [row for row in rows if isinstance(row, dict)]


def human_canonical_to_uniprot_acc(
    canonical_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
    omaid_fallback: str | None = None,
    entry_nr_fallback: int | str | None = None,
    request_sleep_seconds: float = 0.2,
) -> str:
    """Map human OMA identifiers to UniProt. Some SwissProt mnemonics 404 ``…/xref/``."""

    canon = canonical_id.strip() if canonical_id else ""
    if not canon:
        return ""
    if _PRIMARY_UNIPROT_ACC.fullmatch(canon):
        with _HUMAN_SYM_CACHE_GUARD:
            _human_u_cache[canon] = canon
        return canon
    with _HUMAN_SYM_CACHE_GUARD:
        if canon in _human_u_cache and not refresh:
            return _human_u_cache[canon]

    probes_ordered: list[tuple[str, str]] = [(canon, "canon")]
    if omaid_fallback:
        alt = str(omaid_fallback).strip()
        if alt and alt not in {row[0] for row in probes_ordered}:
            probes_ordered.append((alt, "omaid"))
    if entry_nr_fallback is not None and str(entry_nr_fallback).strip():
        try:
            nr_s = str(int(entry_nr_fallback))
        except (TypeError, ValueError):
            nr_s = ""
        if nr_s.isdigit():
            existing = [p[0] for p in probes_ordered]
            if nr_s not in existing:
                probes_ordered.append((nr_s, "entry_nr"))

    picked = ""
    for probe, pk in probes_ordered:
        rows = _fetch_human_xrefs_for_probe(
            probe,
            pk,
            cache_dir=cache_dir,
            refresh=refresh,
            request_sleep_seconds=request_sleep_seconds,
        )
        picked = _human_uniprot_from_xref_rows(rows)
        if picked:
            break

    with _HUMAN_SYM_CACHE_GUARD:
        _human_u_cache[canon] = picked
    return picked


def _dicty_pair_to_row(
    pair: dict[str, object],
    *,
    ddb_to_oid: dict[str, str],
    allowed_oid: set[str] | None,
    human_uniprot_to_orthodb: dict[str, str],
    human_symbols: dict[str, str],
    cache_dir: Path,
    refresh: bool,
    request_sleep_seconds: float,
    omaid_cache: dict[str, str],
    omaid_lock: Lock,
) -> dict[str, str] | None:
    """One pairwise OMA dicty-human row → TSV dict, or ``None`` when skipped."""

    e1 = pair.get("entry_1")
    e2 = pair.get("entry_2")
    if not isinstance(e1, dict) or not isinstance(e2, dict):
        return None
    sp1 = e1.get("species") if isinstance(e1.get("species"), dict) else {}
    sp2 = e2.get("species") if isinstance(e2.get("species"), dict) else {}
    if str(sp1.get("taxon_id")) != "44689" or str(sp2.get("taxon_id")) != HUMAN_TAXID:
        return None
    omaid = str(e1.get("omaid") or "")
    if not omaid:
        return None
    with omaid_lock:
        if omaid not in omaid_cache:
            omaid_cache[omaid] = omaid_to_pub_gene(
                omaid,
                cache_dir=cache_dir,
                refresh=refresh,
                request_sleep_seconds=request_sleep_seconds,
            )
        ddb = omaid_cache[omaid]
    org_oid = ddb_to_oid.get(ddb, "")
    if not org_oid:
        return None
    if allowed_oid is not None and org_oid not in allowed_oid:
        return None
    hum_canon = str(e2.get("canonicalid") or "")
    hum_nr = e2.get("entry_nr")
    hum_u = human_canonical_to_uniprot_acc(
        hum_canon,
        cache_dir=cache_dir,
        refresh=refresh,
        omaid_fallback=str(e2.get("omaid") or "") or None,
        entry_nr_fallback=hum_nr,
        request_sleep_seconds=request_sleep_seconds,
    )
    if not hum_u:
        return None
    rel = _normalise_rel_type(str(pair.get("rel_type") or ""))
    hum_odb = human_uniprot_to_orthodb.get(hum_u, "")
    hum_sym = human_symbols.get(hum_odb, "")
    omaid_h = str(e2.get("omaid") or "")
    return {
        "organism_orthodb_id": org_oid,
        "organism_query_id": ddb,
        "human_uniprot": hum_u,
        "human_orthodb_id": hum_odb,
        "human_symbol": hum_sym,
        "oma_rel_type": rel,
        "oma_id": omaid_h,
    }


def build_dicty_human_rows_from_pairs(
    *,
    fasta_path: Path,
    pairs: list[dict[str, object]],
    human_uniprot_to_orthodb: dict[str, str],
    human_symbols: dict[str, str],
    cache_dir: Path,
    refresh: bool,
    max_organism_genes: int | None,
    show_progress: bool = True,
    oma_workers: int = 1,
) -> list[dict[str, str]]:
    """Join ``pairs/DICDI/9606/`` tuples with FASTA-backed OrthoDB ids.

    Uses thread pool I/O parallelism when ``oma_workers`` > 1 (xref calls are latency-bound).
    """

    ddb_to_oid = invert_pub_gene_to_orthodb(fasta_path)
    distinct_oids = sorted(set(ddb_to_oid.values()))
    if max_organism_genes is not None:
        allowed_oid = set(distinct_oids[: max_organism_genes])
    else:
        allowed_oid = None

    oma_workers = max(1, min(128, oma_workers))
    # With concurrent requests, shorten per-call delay (fetch_text still sleeps each request).
    rq_sleep = 0.03 if oma_workers > 1 else 0.2

    if show_progress:
        print(
            f"  dicty-human xref join: {oma_workers} worker(s), "
            f"{rq_sleep}s request pacing"
        )

    omaid_cache: dict[str, str] = {}
    omaid_lock = Lock()

    rows_out: list[dict[str, str]] = []

    if oma_workers < 2:
        pair_it = tqdm(
            pairs,
            desc="dicty-human join (xref API)",
            total=len(pairs),
            unit="pair",
            disable=not show_progress,
            mininterval=0.25,
        )
        try:
            for pair in pair_it:
                row = _dicty_pair_to_row(
                    pair,
                    ddb_to_oid=ddb_to_oid,
                    allowed_oid=allowed_oid,
                    human_uniprot_to_orthodb=human_uniprot_to_orthodb,
                    human_symbols=human_symbols,
                    cache_dir=cache_dir,
                    refresh=refresh,
                    request_sleep_seconds=rq_sleep,
                    omaid_cache=omaid_cache,
                    omaid_lock=omaid_lock,
                )
                if row:
                    rows_out.append(row)
                    pair_it.set_postfix(kept=len(rows_out))
        finally:
            pair_it.close()
        return rows_out

    def _worker(p: dict[str, object]) -> dict[str, str] | None:
        try:
            return _dicty_pair_to_row(
                p,
                ddb_to_oid=ddb_to_oid,
                allowed_oid=allowed_oid,
                human_uniprot_to_orthodb=human_uniprot_to_orthodb,
                human_symbols=human_symbols,
                cache_dir=cache_dir,
                refresh=refresh,
                request_sleep_seconds=rq_sleep,
                omaid_cache=omaid_cache,
                omaid_lock=omaid_lock,
            )
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=oma_workers) as executor:
        futs = [executor.submit(_worker, p) for p in pairs]
        iterator = tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"dicty-human join ({oma_workers} thr)",
            unit="pair",
            disable=not show_progress,
            mininterval=0.25,
        )
        kept = 0
        try:
            for fut in iterator:
                row = fut.result()
                if row:
                    rows_out.append(row)
                    kept += 1
                iterator.set_postfix(kept=kept)
        finally:
            iterator.close()
    return rows_out


def _default_oma_workers() -> int:
    """Concurrent OMA xref I/O threads (``pairs_vs_human`` join or ``xref_uniprot`` loop).

    Override with ``ORTHOLOGS_OMA_WORKERS`` or legacy ``OMA_WORKERS``, or align with
    ``ORTHOLOGS_DOCKER_CPUS`` when set (e.g. Docker Compose CPU quota).
    """
    raw = (
        os.environ.get("ORTHOLOGS_OMA_WORKERS")
        or os.environ.get("OMA_WORKERS")
        or os.environ.get("ORTHOLOGS_DOCKER_CPUS")
    )
    if raw:
        try:
            return max(1, min(128, int(raw)))
        except ValueError:
            pass
    cpus = os.cpu_count() or 8
    return max(4, min(64, cpus * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch OMA REST orthologs using UniProt accessions from OrthoDB xrefs, "
            "or organism-specific probes (see config ``oma_queries_from``)."
        )
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=sorted(ORGANISMS),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=OMA_CACHE_DIR,
        help="Directory holding cached per-protein OMA JSON responses.",
    )
    parser.add_argument(
        "--xrefs",
        type=Path,
        default=ORTHODB_GENE_XREFS,
    )
    parser.add_argument(
        "--human-xrefs",
        type=Path,
        default=HUMAN_NCBI_ANNOTATIONS,
    )
    parser.add_argument(
        "--human-gene-info",
        type=Path,
        default=HUMAN_GENE_INFO,
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-fetch every OMA response, even if cached.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Cap the number of organism genes queried (debugging).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (for logs or CI).",
    )
    parser.add_argument(
        "--oma-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Thread pool for OMA xref I/O: dicty-human join (pairs_vs_human) "
            "or parallel per-gene fetches (xref_uniprot). "
            "Default: ORTHOLOGS_OMA_WORKERS, else ORTHOLOGS_DOCKER_CPUS, else min(64, max(4, "
            "CPU count × 2))."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ORGANISMS[args.organism]
    if not cfg.get("oma_available", False):
        print(f"OMA not configured for organism={args.organism}; nothing to do.")
        return 0

    if not args.xrefs.exists():
        dl = f"{ORTHODB_BULK_BASE}/{ORTHODB_GENE_XREFS_DUMP.name}"
        print(
            "Missing OrthoDB gene xref table (~4.7 GB gzip). Need one of:\n"
            f"  {ORTHODB_GENE_XREFS_DUMP}\n"
            f"  {ORTHODB_GENE_XREFS_LEGACY}\n"
            "Download (into the first path):\n"
            f"  curl -L '{dl}' -o '{ORTHODB_GENE_XREFS_DUMP}'",
            file=sys.stderr,
        )
        return 1

    organism_id = cfg["orthodb_id"]
    oma_mode = cfg.get("oma_queries_from") or "xref_uniprot"
    organism_fasta = ORTHODB_CACHE_DIR / "api_cache" / "fasta" / f"{organism_id}.fa"

    print(f"=== OMA REST orthologs for {cfg['label']} ===")
    print(f"Cache: {args.cache_dir}")
    print(f"OMA organism probe mode: {oma_mode}")
    show_progress = not args.no_progress

    oma_workers_eff = max(
        1,
        min(
            128,
            (
                args.oma_workers
                if args.oma_workers is not None
                else _default_oma_workers()
            ),
        ),
    )

    rows: list[dict[str, str]] = []
    error_log = args.cache_dir / "_errors.jsonl"
    queried = 0
    skipped_no_probe = 0
    no_orthologs = 0
    n_with_probe = 0
    pair_genomes_meta: tuple[str, str] | None = None
    n_pairs_raw = 0

    print("Resolving human UniProt → OrthoDB id...")
    human_uniprot_to_orthodb = load_human_uniprot_to_orthodb(
        args.xrefs,
        cache_path=XREF_CACHE_DIR / "human_uniprot_to_orthodb.tsv",
        refresh=args.refresh_cache,
    )
    print(f"Human UniProt → OrthoDB ids: {len(human_uniprot_to_orthodb)}")

    print("Loading human OrthoDB id → gene symbol (NCBI gene_info bridge)...")
    ensure_human_gene_info(args.human_gene_info, HUMAN_GENE_INFO_URL)
    ensure_human_ncbi_annotations(args.human_xrefs, args.xrefs)
    human_symbols = load_human_symbols(args.human_xrefs, args.human_gene_info)
    print(f"Human OrthoDB → symbol coverage: {len(human_symbols)}")

    if oma_mode == "pairs_vs_human":
        tup = cfg.get("oma_pair_genomes")
        if not tup or len(tup) != 2:
            print("config error: pairs_vs_human requires ``oma_pair_genomes: (g1, g2)``")
            return 1
        g1, g2 = str(tup[0]), str(tup[1])
        pair_genomes_meta = (g1, g2)
        if not organism_fasta.exists():
            print(f"Missing OrthoDB FASTA: {organism_fasta}")
            return 1
        pub_map = map_to_pub_gene_ids(organism_fasta)
        n_with_probe = sum(1 for accs in pub_map.values() if accs)
        print(f"Dicty FASTA genes with pub_gene_id: {n_with_probe}")
        print(f"Downloading OMA pairwise mappings {g1} vs {g2} …")
        pairs_blob = fetch_all_pairs(
            g1, g2, cache_dir=args.cache_dir, refresh=args.refresh_cache,
            show_progress=show_progress,
        )
        n_pairs_raw = len(pairs_blob)
        print(f"  pairwise rows: {n_pairs_raw}")
        rows = build_dicty_human_rows_from_pairs(
            fasta_path=organism_fasta,
            pairs=pairs_blob,
            human_uniprot_to_orthodb=human_uniprot_to_orthodb,
            human_symbols=human_symbols,
            cache_dir=args.cache_dir,
            refresh=args.refresh_cache,
            max_organism_genes=args.max_genes,
            show_progress=show_progress,
            oma_workers=oma_workers_eff,
        )
        queried = n_pairs_raw
    elif oma_mode == "xref_uniprot":
        print("Resolving organism OrthoDB ids → UniProt via OrthoDB xrefs...")
        organism_to_probe = map_to_uniprot(
            args.organism,
            organism_id,
            xrefs_path=args.xrefs,
            refresh=args.refresh_cache,
        )
        n_with_probe = sum(1 for accs in organism_to_probe.values() if accs)
        print(f"Organism genes with at least one UniProt xref: {n_with_probe}")

        organism_genes_sorted = sorted(organism_to_probe)
        if args.max_genes is not None:
            organism_genes_sorted = organism_genes_sorted[: args.max_genes]

        rq_sleep_x = 0.03 if oma_workers_eff > 1 else 0.2
        if show_progress:
            print(
                f"  OMA xref_uniprot: {oma_workers_eff} worker(s), "
                f"{rq_sleep_x}s request pacing"
            )

        error_log_lock: Lock | None = Lock() if oma_workers_eff > 1 else None

        xref_kw = dict(
            probes_by_gene=organism_to_probe,
            cache_dir=args.cache_dir,
            refresh=args.refresh_cache,
            error_log=error_log,
            human_uniprot_to_orthodb=human_uniprot_to_orthodb,
            human_symbols=human_symbols,
            request_sleep_seconds=rq_sleep_x,
            error_log_lock=error_log_lock,
        )

        n_genes = len(organism_genes_sorted)
        if oma_workers_eff < 2:
            gene_it = tqdm(
                organism_genes_sorted,
                desc="OMA per-gene (UniProt→ortholog JSON)",
                unit="gene",
                disable=not show_progress,
                mininterval=0.25,
            )
            try:
                for i, organism_gene in enumerate(gene_it):
                    _, gene_rows, qd, skipped = _xref_uniprot_process_one_gene(
                        i,
                        organism_gene,
                        **xref_kw,
                    )
                    queried += qd
                    if skipped:
                        skipped_no_probe += 1
                    elif not gene_rows:
                        no_orthologs += 1
                    rows.extend(gene_rows)
                    gene_it.set_postfix(rows=len(rows), queries=queried)
            finally:
                gene_it.close()
        else:
            idx_results: dict[int, tuple[list[dict[str, str]], int, bool]] = {}
            queries_done = 0
            pair_rows_done = 0
            with ThreadPoolExecutor(max_workers=oma_workers_eff) as ex:
                futs = [
                    ex.submit(
                        _xref_uniprot_process_one_gene,
                        gi,
                        organism_genes_sorted[gi],
                        **xref_kw,
                    )
                    for gi in range(n_genes)
                ]
                gene_it = tqdm(
                    as_completed(futs),
                    total=n_genes,
                    desc=f"OMA per-gene ({oma_workers_eff} thr)",
                    unit="gene",
                    disable=not show_progress,
                    mininterval=0.25,
                )
                try:
                    for fut in gene_it:
                        idx, gene_rows, qd, skipped = fut.result()
                        idx_results[idx] = (gene_rows, qd, skipped)
                        queries_done += qd
                        pair_rows_done += len(gene_rows)
                        gene_it.set_postfix(
                            genes=len(idx_results),
                            queries=queries_done,
                            rows=pair_rows_done,
                        )
                finally:
                    gene_it.close()

            for idx in range(n_genes):
                gene_rows, qd, skipped = idx_results[idx]
                queried += qd
                if skipped:
                    skipped_no_probe += 1
                elif not gene_rows:
                    no_orthologs += 1
                rows.extend(gene_rows)
    else:
        print(f"Unknown oma_queries_from mode: {oma_mode!r}")
        return 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates().reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = args.output_dir / f"01d_oma_orthologs_{args.organism}.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)

    rel_counts = df["oma_rel_type"].value_counts().to_dict() if not df.empty else {}
    n_unique_organism = int(df["organism_orthodb_id"].nunique()) if not df.empty else 0
    n_with_symbol = int(df["human_symbol"].astype(bool).sum()) if not df.empty else 0

    manifest = {
        "step": "01d_oma_rest",
        "organism": args.organism,
        "organism_orthodb_id": organism_id,
        "oma_queries_from": oma_mode,
        "oma_pair_genomes": list(pair_genomes_meta) if pair_genomes_meta else None,
        "oma_pair_rows": n_pairs_raw if oma_mode == "pairs_vs_human" else None,
        "organism_fasta": str(organism_fasta)
        if oma_mode == "pairs_vs_human"
        else None,
        "oma_api_base": OMA_API_BASE,
        "cache_dir": str(args.cache_dir),
        "oma_workers": oma_workers_eff,
        "queried_proteins": queried,
        "organism_genes_with_probe": n_with_probe,
        "organism_genes_skipped_no_probe": skipped_no_probe,
        "organism_genes_no_human_ortholog": no_orthologs,
        "n_total_human_pairs": int(len(df)),
        "n_unique_organism_genes_with_human_orthologs": n_unique_organism,
        "n_pairs_with_resolved_symbol": n_with_symbol,
        "oma_rel_type_counts": rel_counts,
        "human_symbol_source": {
            "human_gene_info_path": str(args.human_gene_info),
            "human_gene_info_url": HUMAN_GENE_INFO_URL,
            "human_gene_info_sha256": (
                sha256_file(args.human_gene_info)
                if args.human_gene_info.exists()
                else None
            ),
            "human_xrefs_path": str(args.human_xrefs),
            "human_xrefs_note": HUMAN_NCBI_ANNOTATIONS_NOTE,
            "human_xrefs_sha256": (
                sha256_file(args.human_xrefs) if args.human_xrefs.exists() else None
            ),
        },
        "errors_log": str(error_log),
        "output_tsv": str(tsv_out),
        "generated_utc": utc_now(),
    }
    write_json(args.output_dir / f"01d_oma_orthologs_{args.organism}_manifest.json", manifest)

    print(f"Wrote {len(df)} OMA ortholog pairs to {tsv_out}")
    print(json.dumps(rel_counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
