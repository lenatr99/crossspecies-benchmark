"""Step 01b: OrthoDB orthogroup-membership view of organism-human orthology.

Consumes the bulk OrthoDB flat files from ``data.orthodb.org/current/download/odb_data_dump/``
(OrthoDB tags these ``odb12v2_*`` — OG→genes mapping + levels metadata)
and, for each organism
gene at the chosen taxonomic level (Eukaryota=2759 for Dicty, Metazoa=33208
for C. elegans), records:

  * ``og_id``                — the OrthoDB orthogroup at that level
  * ``human_orthodb_ids``    — semicolon-joined human members of the same OG
  * ``n_organism_members``   — count of organism genes in the OG at that level
  * ``n_human_members``      — count of human genes in the OG at that level
  * ``category``             — {"1:1", "1:many", "many:1", "many:many"}

This is one of four independent evidence sources fed into the consensus step.

The multi‑GB ``*_OG2genes.tab.gz`` is streamed once and filtered by species
prefix (9606_1, 352472_0, 6239_0). The OrthoDB level taxid is decoded from
the OG id format ``<n>at<taxid>`` (e.g. ``9at2759``); ``*_levels.tab.gz``
is consulted only for human-readable level metadata that lands in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

from _utils import (
    cached_download,
    og_level_taxid,
    read_json,
    utc_now,
    write_json,
)
from config import (
    HUMAN_ORTHODB_ID,
    ORGANISMS,
    ORTHODB_BULK_BASE,
    ORTHODB_BULK_FILE_TAG,
    ORTHODB_LEVELS,
    ORTHODB_OG2GENES,
    ORTHODB_RELEASE,
    RESULTS_DIR,
)


BULK_FILES = {
    "OG2genes": ORTHODB_OG2GENES,
    "levels": ORTHODB_LEVELS,
}


def ensure_bulk_files(refresh: bool = False) -> dict[str, str]:
    """Download missing OrthoDB bulk files; return ``{name: sha256}``."""

    base = ORTHODB_BULK_BASE.rstrip("/")
    digests: dict[str, str] = {}
    print(
        "OrthoDB bulk downloads (first run ~4.5 GiB: OG2genes; levels is tiny). "
        "Progress bars via tqdm. Cached under bulk/ — re-runs skip.\n",
        flush=True,
    )
    for name, target in BULK_FILES.items():
        url = f"{base}/{ORTHODB_BULK_FILE_TAG}_{name}.tab.gz"
        cached_download(url, target, refresh=refresh, binary=True, download_label=name)
        metadata_path = target.with_suffix(target.suffix + ".metadata.json")
        digests[name] = (
            read_json(metadata_path).get("sha256", "unknown")
            if metadata_path.exists()
            else "unknown"
        )
    return digests


def parse_levels(levels_path: Path) -> dict[str, dict[str, str]]:
    """Parse the small ``levels.tab.gz`` metadata file.

    Schema (OrthoDB v12): ``level_taxid<TAB>level_name<TAB>n_genes<TAB>n_OGs<TAB>n_species``.
    Returns ``{level_taxid: {"name": ..., "n_genes": ..., "n_ogs": ..., "n_species": ...}}``.
    """

    out: dict[str, dict[str, str]] = {}
    with gzip.open(levels_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            taxid = parts[0]
            out[taxid] = {
                "name": parts[1] if len(parts) > 1 else "",
                "n_genes": parts[2] if len(parts) > 2 else "",
                "n_ogs": parts[3] if len(parts) > 3 else "",
                "n_species": parts[4] if len(parts) > 4 else "",
            }
    return out


def build_og_membership(
    og2genes_path: Path,
    *,
    species_ids: Iterable[str],
    target_level: str,
) -> dict[str, dict[str, list[str]]]:
    """Stream ``OG2genes`` and collect ``{og_id: {species_id: [orthodb_gene_ids]}}``.

    Only OGs whose level taxid matches ``target_level`` are retained. Genes
    outside ``species_ids`` are skipped to keep the in-memory map small.
    """

    species_set = set(species_ids)
    membership: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with gzip.open(og2genes_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            og_id, gene_id = parts
            if og_level_taxid(og_id) != target_level:
                continue
            species_id, _, _ = gene_id.partition(":")
            if species_id not in species_set:
                continue
            membership[og_id][species_id].append(gene_id)
    return membership


def build_og_memberships(
    og2genes_path: Path,
    *,
    organisms: Iterable[str],
    human_id: str,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Stream ``OG2genes`` once and collect memberships for all requested organisms."""

    organism_list = list(organisms)
    by_level: dict[str, list[str]] = defaultdict(list)
    relevant_species_by_organism = {}
    for organism in organism_list:
        cfg = ORGANISMS[organism]
        by_level[cfg["og_level"]].append(organism)
        relevant_species_by_organism[organism] = {cfg["orthodb_id"], human_id}

    memberships: dict[str, dict[str, dict[str, list[str]]]] = {
        organism: defaultdict(lambda: defaultdict(list)) for organism in organism_list
    }
    total = og2genes_path.stat().st_size if og2genes_path.exists() else None
    with og2genes_path.open("rb") as raw:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="scan OG2genes",
            dynamic_ncols=True,
        ) as pbar:
            with gzip.open(raw, "rt", encoding="utf-8", errors="replace") as handle:
                last_pos = raw.tell()
                for line in handle:
                    pos = raw.tell()
                    if pos > last_pos:
                        pbar.update(pos - last_pos)
                        last_pos = pos
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    og_id, gene_id = parts
                    organisms_for_level = by_level.get(og_level_taxid(og_id))
                    if not organisms_for_level:
                        continue
                    species_id, _, _ = gene_id.partition(":")
                    for organism in organisms_for_level:
                        if species_id not in relevant_species_by_organism[organism]:
                            continue
                        memberships[organism][og_id][species_id].append(gene_id)
    return memberships


def og_cardinality(
    og: dict[str, list[str]],
    *,
    organism_id: str,
    human_id: str,
) -> tuple[int, int, str]:
    n_org = len(og.get(organism_id, []))
    n_hum = len(og.get(human_id, []))
    if n_org == 1 and n_hum == 1:
        category = "1:1"
    elif n_org == 1 and n_hum > 1:
        category = "1:many"
    elif n_org > 1 and n_hum == 1:
        category = "many:1"
    else:
        category = "many:many"
    return n_org, n_hum, category


def build_per_organism_table(
    membership: dict[str, dict[str, list[str]]],
    *,
    organism_id: str,
    human_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for og_id, species_genes in membership.items():
        organism_members = species_genes.get(organism_id, [])
        human_members = species_genes.get(human_id, [])
        if not organism_members or not human_members:
            continue
        n_org, n_hum, category = og_cardinality(
            species_genes, organism_id=organism_id, human_id=human_id
        )
        for organism_gene in organism_members:
            rows.append(
                {
                    "organism_orthodb_id": organism_gene,
                    "og_id": og_id,
                    "n_organism_members": n_org,
                    "n_human_members": n_hum,
                    "category": category,
                    "human_orthodb_ids": ";".join(sorted(human_members)),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["organism_orthodb_id", "og_id"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OrthoDB orthogroup membership view per organism."
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=[*sorted(ORGANISMS), "all"],
        help="Organism to process, or 'all' to scan OG2genes once and write every organism.",
    )
    parser.add_argument("--human-id", default=HUMAN_ORTHODB_ID)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download the OrthoDB bulk files even if cached.",
    )
    return parser.parse_args()


def write_organism_outputs(
    *,
    organism: str,
    membership: dict[str, dict[str, list[str]]],
    human_id: str,
    level_info: dict[str, str],
    digests: dict[str, str],
    output_dir: Path,
) -> None:
    organism_cfg = ORGANISMS[organism]
    organism_id = organism_cfg["orthodb_id"]
    target_level = organism_cfg["og_level"]

    print(f"OGs at level {target_level} containing requested species: {len(membership)}")
    df = build_per_organism_table(
        membership,
        organism_id=organism_id,
        human_id=human_id,
    )
    print(
        "Organism genes with at least one shared OG: "
        f"{df['organism_orthodb_id'].nunique() if not df.empty else 0}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = output_dir / f"01b_orthodb_og_{organism}.parquet"
    tsv_out = output_dir / f"01b_orthodb_og_{organism}.tsv"
    df.to_parquet(parquet_out, index=False)
    df.to_csv(tsv_out, sep="\t", index=False)

    n_one_to_one = int((df["category"] == "1:1").sum()) if not df.empty else 0

    manifest = {
        "step": "01b_orthodb_og",
        "orthodb_release": ORTHODB_RELEASE,
        "orthodb_bulk_file_tag": ORTHODB_BULK_FILE_TAG,
        "orthodb_bulk_base": ORTHODB_BULK_BASE,
        "organism": organism,
        "organism_orthodb_id": organism_id,
        "human_orthodb_id": human_id,
        "level_taxid": target_level,
        "level_metadata": level_info,
        "input_sha256": digests,
        "output_parquet": str(parquet_out),
        "output_tsv": str(tsv_out),
        "n_ogs_with_both_species": len(membership),
        "n_organism_genes": int(df["organism_orthodb_id"].nunique()) if not df.empty else 0,
        "n_one_to_one_ogs": n_one_to_one,
        "generated_utc": utc_now(),
    }
    write_json(output_dir / f"01b_orthodb_og_{organism}.manifest.json", manifest)

    print(f"Wrote: {parquet_out}")
    print(f"Wrote: {tsv_out}")
    print(f"1:1 OG entries: {n_one_to_one}")


def organism_outputs_current(
    *,
    organism: str,
    human_id: str,
    level_info: dict[str, str],
    digests: dict[str, str],
    output_dir: Path,
) -> bool:
    """Return True when existing 01b outputs match the current bulk inputs."""

    cfg = ORGANISMS[organism]
    parquet_out = output_dir / f"01b_orthodb_og_{organism}.parquet"
    tsv_out = output_dir / f"01b_orthodb_og_{organism}.tsv"
    manifest_path = output_dir / f"01b_orthodb_og_{organism}.manifest.json"
    if not (parquet_out.exists() and tsv_out.exists() and manifest_path.exists()):
        return False

    try:
        manifest = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return False

    expected = {
        "step": "01b_orthodb_og",
        "orthodb_release": ORTHODB_RELEASE,
        "orthodb_bulk_file_tag": ORTHODB_BULK_FILE_TAG,
        "orthodb_bulk_base": ORTHODB_BULK_BASE,
        "organism": organism,
        "organism_orthodb_id": cfg["orthodb_id"],
        "human_orthodb_id": human_id,
        "level_taxid": cfg["og_level"],
        "level_metadata": level_info,
        "input_sha256": digests,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    organisms = sorted(ORGANISMS) if args.organism == "all" else [args.organism]

    labels = ", ".join(ORGANISMS[organism]["label"] for organism in organisms)
    print(f"=== OrthoDB OG membership for {labels} ===")
    print(
        "Level taxids: "
        + ", ".join(f"{organism}={ORGANISMS[organism]['og_level']}" for organism in organisms)
    )
    print(f"Bulk dir: {BULK_FILES['OG2genes'].parent}")

    digests = ensure_bulk_files(refresh=args.refresh_cache)

    levels_meta = parse_levels(BULK_FILES["levels"])
    for organism in organisms:
        target_level = ORGANISMS[organism]["og_level"]
        print(f"Level metadata [{organism}]: {levels_meta.get(target_level, {'name': 'unknown'})}")

    current = [
        organism_outputs_current(
            organism=organism,
            human_id=args.human_id,
            level_info=levels_meta.get(ORGANISMS[organism]["og_level"], {"name": "unknown"}),
            digests=digests,
            output_dir=args.output_dir,
        )
        for organism in organisms
    ]
    if all(current) and not args.refresh_cache:
        print("All requested 01b outputs are current; skipping OG2genes scan.")
        return 0

    print("Streaming OG2genes once for all requested organisms...")
    memberships = build_og_memberships(
        BULK_FILES["OG2genes"],
        organisms=organisms,
        human_id=args.human_id,
    )

    for organism in organisms:
        print(f"\nWriting outputs for {organism}...")
        target_level = ORGANISMS[organism]["og_level"]
        write_organism_outputs(
            organism=organism,
            membership=memberships[organism],
            human_id=args.human_id,
            level_info=levels_meta.get(target_level, {"name": "unknown"}),
            digests=digests,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
