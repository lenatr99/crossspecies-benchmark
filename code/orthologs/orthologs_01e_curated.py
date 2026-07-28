"""Step 01e: curated literature orthologs (fourth evidence source).

Two organism-specific curated tables, treated as ground truth where available:

  * **C. elegans — OrthoList** (Kim et al. 2018, NAR; master ``.xlsx`` from OrthoList
    ``send_master``, e.g. ``http://ortholist.shaye-lab.org/send_master``). A consensus set of
    C. elegans/human orthologs derived from four independent prediction
    methods, manually reviewed. Outperforms any single algorithmic source
    on benchmarked gene sets.

  * **Dictyostelium — dictyBase**: the general release file
    ``ortholog_information.txt`` lists per-gene orthologs from several
    sources (e.g. OrthoMCL, InParanoid) with ``H.sapiens:ENSP…`` style human
    peptide ids. We map Ensembl proteins to UniProt (cached) and then map
    UniProt → HGNC symbols via the same OrthoDB + NCBI ``gene_info`` bridge
    used elsewhere in the pipeline. Older two-column hand-curated tables
    are still accepted.
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

from _utils import (
    ensure_human_gene_info,
    ensure_human_ncbi_annotations,
    load_human_symbols,
    load_human_uniprot_to_orthodb,
    sha256_file,
    utc_now,
    write_json,
)
from config import (
    DICTYBASE_CURATED_PATH,
    DICTYBASE_ENSP_SYMBOL_CACHE,
    DICTYBASE_ENSP_UNIPROT_CACHE,
    DICTYBASE_ORTHOLOG_INFO_URL,
    HUMAN_GENE_INFO,
    HUMAN_GENE_INFO_URL,
    HUMAN_NCBI_ANNOTATIONS,
    ORGANISMS,
    ORTHODB_CACHE_DIR,
    ORTHODB_GENE_XREFS,
    ORTHOLIST2_PATH,
    RESULTS_DIR,
)

_HUMAN_UNIPROT_ORTHODB_CACHE = ORTHODB_CACHE_DIR / "xrefs" / "human_uniprot_to_orthodb.tsv"

UNIPROT_API = "https://rest.uniprot.org"
ENSEMBL_REST = "https://rest.ensembl.org"
ENSEMBL_UA = "dictybase-ortholog-pipeline/01e (dicty-human ortholog curation; research use)"


# Primary: OrthoList master workbook (``.xlsx``) served by OrthoList service.
ORTHOLIST2_DOWNLOAD_URLS: tuple[str, ...] = (
    "http://ortholist.shaye-lab.org/send_master",
    # Legacy comma-separated supplements (alternate layouts):
    "https://greenwaldlab.cb.columbia.edu/orthlist2/orthlist2.csv",
    "https://greenwaldlab.cb.columbia.edu/orthlist2/OrthoList_2_master.csv",
)


def _maybe_download(url: str, target: Path) -> bool:
    """Download ``url`` to ``target`` if missing. Return True on success."""

    if target.exists():
        return True
    try:
        from _utils import cached_download

        cached_download(url, target, refresh=False, binary=True)
        return target.exists() and target.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        print(f"Could not auto-download {url}: {exc}")
        return False


def _maybe_download_text(url: str, target: Path) -> bool:
    """Download UTF-8 ``url`` to ``target`` if missing (text/table responses)."""

    if target.exists():
        return True
    try:
        from _utils import cached_download

        cached_download(url, target, refresh=False, binary=False)
        return target.exists() and target.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        print(f"Could not auto-download {url}: {exc}")
        return False


def _canonical_header_token(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(header).strip().lower())


def _pick_ortholist_wormbase_column(columns: list[str]) -> str | None:
    keyed = {_canonical_header_token(h): h for h in columns}
    for cand in ("wormbaseid", "wormbasegeneid", "wbgeneid", "wormbasegene"):
        if cand in keyed:
            return keyed[cand]
    for h in columns:
        t = _canonical_header_token(h)
        if t == "wormbase" or ("wormbase" in t and "id" in t):
            return h
        if "wbgene" in t:
            return h
    return None


def _pick_ortholist_hgnc_column(columns: list[str]) -> str | None:
    keyed = {_canonical_header_token(h): h for h in columns}
    for cand in ("hgncsymbol", "hgnc", "genesymbol"):
        if cand in keyed:
            return keyed[cand]
    for h in columns:
        t = _canonical_header_token(h)
        if t in {"symbol", "humansymbol", "gene"}:
            return h
        if "hgnc" in t and ("symbol" in t or "sym" in t):
            return h
    return None


def parse_orthlist2(path: Path) -> pd.DataFrame:
    """Parse OrthoList master workbook or CSV → ``[wbgene_id, human_symbol, source]`` rows.

    Accepts the Shaye lab ``ortholist_master.xlsx`` (``WormBase ID``, ``HGNC Symbol``, …),
    comma-separated supplementary tables from older URLs, or any table with recognizable
    WormBase gene id and HGNC symbol columns.
    """

    if not path.exists():
        return pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    elif suf in (".xlsx", ".xlsm", ".xltx"):
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    elif suf == ".xls":
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(
            f"Unsupported OrthoList file type for {path.name}; "
            "use .csv, .xlsx, or OrthoList auto-download URLs."
        )
    df.columns = [str(c).strip() for c in df.columns]

    wb_col = _pick_ortholist_wormbase_column(list(df.columns))
    hg_col = _pick_ortholist_hgnc_column(list(df.columns))
    if wb_col is None or hg_col is None:
        raise ValueError(
            f"OrthoList file {path} missing required columns; got {list(df.columns)}. "
            "Expected one WormBase id column and one HGNC symbol column."
        )

    out = (
        df[[wb_col, hg_col]]
        .rename(columns={wb_col: "organism_gene_id", hg_col: "human_symbol"})
        .dropna()
    )
    out["organism_gene_id"] = out["organism_gene_id"].str.strip()
    out["human_symbol"] = out["human_symbol"].str.strip()
    out = out[(out["organism_gene_id"] != "") & (out["human_symbol"] != "")]
    out["source"] = "OrthoList2"
    return out.drop_duplicates().reset_index(drop=True)


def _norm_ensembl_protein(ident: str) -> str:
    ident = ident.strip()
    if ident.upper().startswith("ENSP") and "." in ident:
        return ident.split(".", 1)[0]
    return ident


def _species_token_is_human(species: str) -> bool:
    """Match ``H.sapiens:`` / ``H. sapiens:`` style species tags in dictyBase ortholog lists."""

    x = re.sub(r"\s+", "", species.strip().lower())
    if x in {"human", "hsap", "hs"}:
        return True
    if not x.startswith("h."):
        return False
    return "sap" in x and "iens" in x


def _load_ensp_uniprot_pairs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                out[parts[0].strip()] = parts[1].strip()
    return out


def _append_ensp_uniprot_pairs(path: Path, pairs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("# Ensembl_Protein\tUniProtKB_AC (UniProt id-mapping cache)\n")
        for ensp, u in sorted(pairs.items()):
            handle.write(f"{ensp}\t{u}\n")


def _append_ensp_symbol_pairs(path: Path, pairs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("# Ensembl_protein\tHGNC_symbol (Ensembl REST lookup cache)\n")
        for ensp, sym in sorted(pairs.items()):
            handle.write(f"{ensp}\t{sym}\n")


def _retry_after_seconds(value: str | None, fallback: float) -> float:
    if not value:
        return fallback
    if value.isdigit():
        return float(value)
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, parsed.timestamp() - time.time()) if parsed else fallback


def _ensembl_post_lookup_hsapiens(ids: list[str]) -> dict[str, object]:
    """``POST /lookup/id/homo_sapiens`` — batch protein/transcript/gene metadata."""

    if not ids:
        return {}
    url = f"{ENSEMBL_REST}/lookup/id/homo_sapiens"
    hdrs = {
        "User-Agent": ENSEMBL_UA,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    backoff = (4.0, 12.0, 30.0, 75.0, 180.0, 300.0)
    last_exc: Exception | None = None
    for attempt in range(len(backoff) + 1):
        try:
            r = requests.post(url, json={"ids": ids}, headers=hdrs, timeout=(30, 300))
            if r.status_code in {429, 500, 502, 503, 504} and attempt < len(backoff):
                time.sleep(_retry_after_seconds(r.headers.get("Retry-After"), backoff[attempt]))
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                return {}
            return data
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt >= len(backoff):
                raise
            time.sleep(backoff[attempt])
    if last_exc:
        raise last_exc
    return {}


def _map_ensp_chunk_to_hgnc_via_ensembl(en_chunk: list[str]) -> dict[str, str]:
    """One chunk: ENSP → translation → ENST → transcript → ENSG → gene ``display_name``."""

    out: dict[str, str] = {}
    raw1 = _ensembl_post_lookup_hsapiens(en_chunk)
    en_to_enst: dict[str, str] = {}
    for eid in en_chunk:
        blob = raw1.get(eid)
        if not isinstance(blob, dict):
            continue
        if blob.get("object_type") != "Translation":
            continue
        parent = blob.get("Parent")
        if isinstance(parent, str) and parent.startswith("ENST"):
            en_to_enst[eid] = parent
    if not en_to_enst:
        return out
    enst_unique = sorted(set(en_to_enst.values()))
    raw2 = _ensembl_post_lookup_hsapiens(enst_unique)
    enst_to_ensg: dict[str, str] = {}
    for enst in enst_unique:
        b = raw2.get(enst)
        if not isinstance(b, dict) or b.get("object_type") != "Transcript":
            continue
        g = b.get("Parent")
        if isinstance(g, str) and g.startswith("ENSG"):
            enst_to_ensg[enst] = g
    if not enst_to_ensg:
        return out
    gene_ids = sorted(set(enst_to_ensg.values()))
    raw3 = _ensembl_post_lookup_hsapiens(gene_ids)
    for ensp, enst in en_to_enst.items():
        g = enst_to_ensg.get(enst)
        if not g:
            continue
        gb = raw3.get(g)
        if not isinstance(gb, dict) or gb.get("object_type") != "Gene":
            continue
        sym = (gb.get("display_name") or "").strip()
        if sym:
            out[ensp] = sym
    return out


def _map_ensp_chunk_to_hgnc_via_ensembl_resilient(
    en_chunk: list[str],
    *,
    depth: int = 0,
) -> dict[str, str]:
    """Map a batch through Ensembl; bisect repeated network failures."""

    try:
        return _map_ensp_chunk_to_hgnc_via_ensembl(en_chunk)
    except requests.RequestException as exc:
        if len(en_chunk) <= 1:
            print(f"  warning: Ensembl lookup failed for {en_chunk[0]}: {exc}", flush=True)
            return {}
        mid = len(en_chunk) // 2
        time.sleep(min(60.0, 2.0 * (depth + 1)))
        left = _map_ensp_chunk_to_hgnc_via_ensembl_resilient(
            en_chunk[:mid],
            depth=depth + 1,
        )
        right = _map_ensp_chunk_to_hgnc_via_ensembl_resilient(
            en_chunk[mid:],
            depth=depth + 1,
        )
        return {**left, **right}


def _ensembl_proteins_to_gene_symbols(
    identifiers: set[str],
    *,
    cache_path: Path,
    batch_size: int = 100,
    show_progress: bool = True,
) -> dict[str, str]:
    """Map ENSP… → HGNC-like gene symbol via Ensembl REST fallback."""

    sym_map: dict[str, str] = _load_ensp_uniprot_pairs(cache_path)
    missing = sorted(identifiers - set(sym_map))
    if not missing:
        return {e: sym_map[e] for e in identifiers if e in sym_map}

    if not show_progress:
        print(
            f"  Ensembl REST: {len(missing)} human peptide id(s) → gene symbol "
            f"(≤{batch_size} ids / batch, {((len(missing) + batch_size - 1) // batch_size)} batches)…",
            flush=True,
        )

    chunks = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    with tqdm(
        chunks,
        desc="01e Ensembl ENSP→HGNC",
        unit="batch",
        total=len(chunks),
        disable=not show_progress,
        mininterval=0.25,
    ) as pbar:
        for chunk in pbar:
            new = _map_ensp_chunk_to_hgnc_via_ensembl_resilient(chunk)
            sym_map.update(new)
            if new:
                _append_ensp_symbol_pairs(cache_path, new)
            pbar.set_postfix(mapped=len(sym_map))
            time.sleep(0.35)

    return {e: sym_map[e] for e in identifiers if e in sym_map}


def _poll_uniprot_mapping_result(
    job_id: str,
    *,
    timeout_sec: float,
) -> str:
    """Block until id-mapping job completes; download TSV results."""

    t0 = time.time()
    hdrs = {"User-Agent": "dictybase-ortholog-pipeline/01e"}
    last_log = t0
    while time.time() - t0 < timeout_sec:
        st = requests.get(
            f"{UNIPROT_API}/idmapping/status/{job_id}",
            headers=hdrs,
            timeout=120,
        )
        st.raise_for_status()
        payload = st.json()
        job_status = str(payload.get("jobStatus") or "").upper()
        if job_status == "FINISHED":
            res = requests.get(
                f"{UNIPROT_API}/idmapping/results/{job_id}",
                params={"format": "tsv"},
                headers=hdrs,
                timeout=300,
            )
            res.raise_for_status()
            return res.text
        if job_status in {"ERROR", "FAILED"}:
            raise RuntimeError(f"UniProt idmapping failed: {payload}")
        # NEW / RUNNING / empty: keep polling
        if time.time() - last_log > 45.0:
            print(
                f"  UniProt id-mapping job {job_id} still {job_status or 'pending'} "
                f"({int(time.time() - t0)}s / {int(timeout_sec)}s cap)…",
                flush=True,
            )
            last_log = time.time()
        time.sleep(2.0)
    raise TimeoutError(f"UniProt idmapping job {job_id!r} did not finish within {timeout_sec:.0f}s")


def _timeout_for_batch(n_ids: int) -> float:
    """Scale wait with batch size (UniProt can be slow under load)."""

    return min(7200.0, max(600.0, 60.0 + float(n_ids) * 8.0))


def _fetch_uniprot_ensembl_batch(
    chunk: list[str],
    *,
    hdrs: dict[str, str],
) -> dict[str, str]:
    """Submit one batch; return From→To map. Raises on HTTP or terminal API error."""

    post = requests.post(
        f"{UNIPROT_API}/idmapping/run",
        headers=hdrs,
        data={
            "from": "Ensembl_Protein",
            "to": "UniProtKB",
            "ids": "\n".join(chunk),
        },
        timeout=180,
    )
    post.raise_for_status()
    job_id = post.json()["jobId"]
    tsv = _poll_uniprot_mapping_result(
        job_id,
        timeout_sec=_timeout_for_batch(len(chunk)),
    )
    pairs: dict[str, str] = {}
    for raw in tsv.strip().splitlines():
        if raw.lower().startswith("from\t"):
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        from_id, to_id = parts[0].strip(), parts[1].strip()
        if ";" in to_id:
            to_id = to_id.split(";", 1)[0].strip()
        pairs[from_id] = to_id
    return pairs


def _fetch_uniprot_ensembl_batch_with_split(
    chunk: list[str],
    *,
    hdrs: dict[str, str],
    depth: int = 0,
) -> dict[str, str]:
    """Map one batch; on timeout bisect and retry (UniProt jobs can exceed short caps)."""

    if not chunk:
        return {}
    max_depth = 14  # 2^14 splits from 380 → single ids
    try:
        return _fetch_uniprot_ensembl_batch(chunk, hdrs=hdrs)
    except TimeoutError:
        if len(chunk) <= 1 or depth >= max_depth:
            one = chunk[0] if chunk else ""
            print(
                f"  WARN: UniProt id-mapping timed out on {one!r} "
                f"(skip {len(chunk)} id(s)).",
                flush=True,
            )
            return {}
        mid = len(chunk) // 2
        left = _fetch_uniprot_ensembl_batch_with_split(
            chunk[:mid], hdrs=hdrs, depth=depth + 1
        )
        right = _fetch_uniprot_ensembl_batch_with_split(
            chunk[mid:], hdrs=hdrs, depth=depth + 1
        )
        merged = {**left, **right}
        return merged


def _ensembl_proteins_to_uniprot(
    identifiers: set[str],
    *,
    cache_path: Path,
    batch_size: int = 200,
    show_progress: bool = True,
) -> dict[str, str]:
    """Map Ensembl peptide ids (ENSP…) → UniProt accessions (batched, cached on disk)."""

    ident_map: dict[str, str] = _load_ensp_uniprot_pairs(cache_path)
    missing = sorted(identifiers - set(ident_map))
    if not missing:
        return {k: ident_map[k] for k in identifiers}

    hdrs = {
        "User-Agent": "dictybase-ortholog-pipeline/01e",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    if not show_progress:
        print(
            f"  UniProt id-mapping: {len(missing)} Ensembl protein id(s) "
            f"in batches of ≤{batch_size} (with split-on-timeout)…",
            flush=True,
        )

    batches: list[list[str]] = []
    start = 0
    while start < len(missing):
        batches.append(missing[start : start + batch_size])
        start += batch_size

    with tqdm(
        batches,
        desc="01e UniProt ENSP→AC",
        unit="batch",
        total=len(batches),
        disable=not show_progress,
        mininterval=0.25,
    ) as pbar:
        for chunk in pbar:
            pairs = _fetch_uniprot_ensembl_batch_with_split(chunk, hdrs=hdrs)
            ident_map.update(pairs)
            _append_ensp_uniprot_pairs(cache_path, pairs)
            pbar.set_postfix(ids=len(ident_map))
            time.sleep(1.0)

    return {k: ident_map[k] for k in identifiers if k in ident_map}


def parse_dictybase_simple_two_col(path: Path) -> pd.DataFrame:
    """Legacy: ``DDB_Gxxx<TAB>HumanSymbol`` table with optional header sniffing."""

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = "ddb_g" in sample.lower() and (
            "human" in sample.lower() or "symbol" in sample.lower()
        ) and any(
            line.lower().split("\t")[0].startswith(("ddb_g_id", "ddb_id", "id"))
            for line in sample.splitlines()[:1]
            if line.strip()
        )
        reader = csv.reader(handle, delimiter="\t")
        if has_header:
            header = next(reader)
            cols = [c.strip().lower() for c in header]
            ddb_idx = next((i for i, c in enumerate(cols) if "ddb" in c), 0)
            sym_idx = next((i for i, c in enumerate(cols) if "symbol" in c or "human" in c), 1)
            for row in reader:
                if len(row) <= max(ddb_idx, sym_idx):
                    continue
                rows.append({
                    "organism_gene_id": row[ddb_idx].strip(),
                    "human_symbol": row[sym_idx].strip(),
                })
        else:
            for row in reader:
                if len(row) < 2:
                    continue
                rows.append({
                    "organism_gene_id": row[0].strip(),
                    "human_symbol": row[1].strip(),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])
    df = df[(df["organism_gene_id"] != "") & (df["human_symbol"] != "")]
    df["source"] = "dictyBase"
    return df.drop_duplicates().reset_index(drop=True)


def parse_dictybase(
    path: Path,
    *,
    orthodb_xrefs: Path,
    human_xrefs: Path,
    human_gene_info: Path,
    ensp_uniprot_cache: Path,
    ensp_symbol_cache: Path,
    uniprot_orthodb_cache: Path,
    refresh_uniprot_bridge: bool = False,
    hgnc_via: str = "ensembl",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Parse dictyBase ortholog export into ``organism_gene_id, human_symbol, source``."""

    if not path.exists():
        return pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])

    sniff = pd.read_csv(path, sep="\t", dtype=str, nrows=2, encoding="utf-8", on_bad_lines="skip")
    cols_norm = [str(c).strip().lower() for c in sniff.columns]

    if "orthologs" not in cols_norm:
        return parse_dictybase_simple_two_col(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    )
    df.columns = [str(c).strip() for c in df.columns]

    try:
        gid_col = next(
            c for c in df.columns
            if str(c).strip().lower().replace(" ", "") == "geneid"
        )
    except StopIteration:
        gid_col = next(c for c in df.columns if "gene" in c.lower() and "id" in c.lower())
    try:
        src_col = next(c for c in df.columns if str(c).strip().lower() == "source")
    except StopIteration:
        src_col = None
    try:
        orth_col = next(c for c in df.columns if str(c).strip().lower() == "orthologs")
    except StopIteration:
        return parse_dictybase_simple_two_col(path)

    pair_rows: list[dict[str, str]] = []
    ensp_need: set[str] = set()

    for _, row in df.iterrows():
        ddb = str(row.get(gid_col, "") or "").strip()
        orth_blob = str(row.get(orth_col, "") or "")
        if not ddb.startswith("DDB_G") or not orth_blob:
            continue
        provenance = ""
        if src_col is not None:
            provenance = str(row.get(src_col, "") or "").strip()
        for chunk in orth_blob.split("|"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            tag, ident = chunk.split(":", 1)
            tag, ident = tag.strip(), ident.strip()
            if not _species_token_is_human(tag):
                continue
            if not ident.upper().startswith("ENSP"):
                continue
            eid = _norm_ensembl_protein(ident)
            ensp_need.add(eid)
            pair_rows.append({
                "ddb": ddb,
                "ensp": eid,
                "subsource": provenance or "dictyBase",
            })

    if not pair_rows:
        return pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])

    if hgnc_via == "ensembl":
        ensp_to_sym = _ensembl_proteins_to_gene_symbols(
            ensp_need,
            cache_path=ensp_symbol_cache,
            show_progress=show_progress,
        )
    else:
        ensp_to_up = _ensembl_proteins_to_uniprot(
            ensp_need,
            cache_path=ensp_uniprot_cache,
            show_progress=show_progress,
        )
        up_to_hoid = load_human_uniprot_to_orthodb(
            orthodb_xrefs,
            cache_path=uniprot_orthodb_cache,
            refresh=refresh_uniprot_bridge,
        )
        ensure_human_gene_info(human_gene_info, HUMAN_GENE_INFO_URL)
        ensure_human_ncbi_annotations(human_xrefs, orthodb_xrefs)
        hoid_to_sym = load_human_symbols(human_xrefs, human_gene_info)
        ensp_to_sym = {}
        for ensp in ensp_need:
            acc = ensp_to_up.get(ensp)
            if not acc:
                continue
            hoid = up_to_hoid.get(acc)
            if not hoid:
                continue
            hs = hoid_to_sym.get(hoid, "").strip()
            if hs:
                ensp_to_sym[ensp] = hs

    out_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pr in pair_rows:
        hs = ensp_to_sym.get(pr["ensp"], "").strip()
        if not hs:
            continue
        sub = pr["subsource"]
        label = f"dictyBase/{sub}" if sub else "dictyBase"
        key = (pr["ddb"], hs, label)
        if key in seen:
            continue
        seen.add(key)
        out_rows.append({
            "organism_gene_id": pr["ddb"],
            "human_symbol": hs,
            "source": label,
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])
    return out.sort_values(["organism_gene_id", "human_symbol"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse curated literature orthologs (OrthoList 2 / dictyBase)."
    )
    parser.add_argument(
        "--organism",
        default="elegans",
        choices=sorted(ORGANISMS),
    )
    parser.add_argument(
        "--orthlist2-path",
        type=Path,
        default=ORTHOLIST2_PATH,
        help="Local path to OrthoList master file (.xlsx from ortholist…/send_master or legacy CSV).",
    )
    parser.add_argument(
        "--dictybase-path",
        type=Path,
        default=DICTYBASE_CURATED_PATH,
        help="Local path to dictyBase ``ortholog_information.txt`` or a 2-column legacy TSV.",
    )
    parser.add_argument(
        "--xrefs",
        type=Path,
        default=ORTHODB_GENE_XREFS,
        help="OrthoDB gene xrefs (used to map human UniProt ↔ OrthoDB id for dicty).",
    )
    parser.add_argument(
        "--human-xrefs",
        type=Path,
        default=HUMAN_NCBI_ANNOTATIONS,
        help="Human OrthoDB annotations (NCBI bridge) for HGNC resolution.",
    )
    parser.add_argument(
        "--human-gene-info",
        type=Path,
        default=HUMAN_GENE_INFO,
    )
    parser.add_argument(
        "--dicty-hgnc-via",
        choices=("ensembl", "uniprot"),
        default="ensembl",
        help=(
            "For dictyBase ENSP… ids: 'ensembl' = batched Ensembl REST "
            "(default); 'uniprot' = UniProt id mapping + local OrthoDB/NCBI "
            "symbol bridge."
        ),
    )
    parser.add_argument(
        "--dicty-ensp-symbol-cache",
        type=Path,
        default=DICTYBASE_ENSP_SYMBOL_CACHE,
        help="Cache Ensembl peptide id → HGNC symbol (ensembl resolver).",
    )
    parser.add_argument(
        "--dicty-ensp-uniprot-cache",
        type=Path,
        default=DICTYBASE_ENSP_UNIPROT_CACHE,
        help="Cache Ensembl peptide → UniProt (uniprot resolver only).",
    )
    parser.add_argument(
        "--human-uniprot-orthodb-cache",
        type=Path,
        default=_HUMAN_UNIPROT_ORTHODB_CACHE,
        help="Cached table built from OrthoDB UniProt xrefs for Homo sapiens.",
    )
    parser.add_argument(
        "--refresh-uniprot-orthodb-cache",
        action="store_true",
        help="Rebuild the human UniProt→OrthoDB cache from --xrefs.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Attempt to download missing curated files from canonical URLs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. dicty Ensembl / UniProt batches).",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ORGANISMS[args.organism]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.output_dir / f"01e_curated_{args.organism}.tsv"

    note: str
    if args.organism == "elegans":
        if args.auto_download and not args.orthlist2_path.exists():
            for url in ORTHOLIST2_DOWNLOAD_URLS:
                if _maybe_download(url, args.orthlist2_path):
                    break
        df = parse_orthlist2(args.orthlist2_path)
        source_path = args.orthlist2_path
        source_label = "OrthoList2"
        note = (
            f"Parsed OrthoList 2 from {source_path}"
            if df.shape[0] > 0
            else f"OrthoList 2 file missing or empty at {source_path}; place CSV manually and rerun."
        )
    elif args.organism == "dicty":
        if args.auto_download and not args.dictybase_path.exists():
            _maybe_download_text(DICTYBASE_ORTHOLOG_INFO_URL, args.dictybase_path)
        df = parse_dictybase(
            args.dictybase_path,
            orthodb_xrefs=args.xrefs,
            human_xrefs=args.human_xrefs,
            human_gene_info=args.human_gene_info,
            ensp_uniprot_cache=args.dicty_ensp_uniprot_cache,
            ensp_symbol_cache=args.dicty_ensp_symbol_cache,
            uniprot_orthodb_cache=args.human_uniprot_orthodb_cache,
            refresh_uniprot_bridge=args.refresh_uniprot_orthodb_cache,
            hgnc_via=args.dicty_hgnc_via,
            show_progress=not args.no_progress,
        )
        source_path = args.dictybase_path
        source_label = "dictyBase"
        if not source_path.exists():
            note = (
                f"No dictyBase ortholog file at {source_path}. "
                "Run with --auto-download or place dictyBase ``ortholog_information.txt`` manually."
            )
        elif df.empty:
            note = (
                f"dictyBase file present at {source_path} but no human HGNC rows were produced "
                f"(resolver={args.dicty_hgnc_via}; check ENSP identifiers and network). "
                "An empty curated TSV is written."
            )
        else:
            note = (
                f"Parsed dictyBase orthologs ({args.dicty_hgnc_via}) from {source_path.name}."
            )
    else:
        df = pd.DataFrame(columns=["organism_gene_id", "human_symbol", "source"])
        source_path = None
        source_label = None
        note = f"No curated source configured for organism={args.organism}."

    df.to_csv(out_tsv, sep="\t", index=False)

    manifest = {
        "step": "01e_curated",
        "organism": args.organism,
        "label": cfg["label"],
        "source": source_label,
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": sha256_file(source_path) if source_path is not None and source_path.exists() else None,
        "n_rows": int(len(df)),
        "n_unique_organism_genes": int(df["organism_gene_id"].nunique()) if not df.empty else 0,
        "n_unique_human_symbols": int(df["human_symbol"].nunique()) if not df.empty else 0,
        "output_tsv": str(out_tsv),
        "note": note,
        "generated_utc": utc_now(),
    }
    if args.organism == "dicty":
        manifest["dictybase_download_url"] = DICTYBASE_ORTHOLOG_INFO_URL
        manifest["dicty_hgnc_via"] = args.dicty_hgnc_via
        manifest["dicty_ensp_symbol_cache"] = str(args.dicty_ensp_symbol_cache)
        manifest["dicty_ensp_uniprot_cache"] = str(args.dicty_ensp_uniprot_cache)
        manifest["human_uniprot_orthodb_cache"] = str(args.human_uniprot_orthodb_cache)

    write_json(args.output_dir / f"01e_curated_{args.organism}_manifest.json", manifest)

    print(note)
    print(f"Wrote {len(df)} curated rows to {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
