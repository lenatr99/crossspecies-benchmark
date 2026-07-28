"""Step 01a: OrthoDB pair-API ortholog snapshot (one of four evidence sources).

This script fetches organism-to-human ortholog *pairs* from the OrthoDB v12
``/orthologs`` REST endpoint and writes per-organism CSV summaries that the
legacy pipeline used as its sole input. In the new consensus pipeline these
CSVs are kept as a sanity-check / human-readable view; the consensus
(``orthologs_02_consensus.py``) draws orthogroup membership from the bulk
download via ``orthologs_01b_orthodb_og.py`` instead.

The live OrthoDB API is contacted only when ``--refresh-cache`` is passed or
when a required cache file is missing. Re-runs replay the JSON/FASTA cache
under ``data/orthodb/odb12v1`` so manuscript outputs do not silently change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config import (
    HUMAN_ORTHODB_ID,
    ORGANISMS,
    ORTHODB_API_VERSION,
    ORTHODB_CACHE_DIR,
    ORTHODB_RELEASE,
    RESULTS_DIR,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fetch_text(url: str, params: dict[str, str], timeout: int = 120) -> str:
    time.sleep(1)
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


def cache_metadata(url: str, params: dict[str, str], content: str) -> dict[str, Any]:
    return {
        "orthodb_release": ORTHODB_RELEASE,
        "api_version": ORTHODB_API_VERSION,
        "url": url,
        "params": params,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_text(content),
    }


def load_or_fetch_text(
    cache_path: Path,
    metadata_path: Path,
    url: str,
    params: dict[str, str],
    refresh_cache: bool,
) -> str:
    if cache_path.exists() and not refresh_cache:
        return cache_path.read_text(encoding="utf-8")

    if cache_path.exists() and refresh_cache:
        print(f"Refreshing cached OrthoDB input: {cache_path}")
    else:
        print(f"Missing cached OrthoDB input, downloading: {cache_path}")

    content = fetch_text(url, params)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content, encoding="utf-8")
    write_json(metadata_path, cache_metadata(url, params, content))
    return content


def get_all_genes_in_species(
    species_id: str,
    cache_dir: Path,
    refresh_cache: bool = False,
) -> int:
    """Count genes in a species from a cached OrthoDB FASTA response."""
    url = f"https://data.orthodb.org/{ORTHODB_API_VERSION}/fasta"
    params = {"species": species_id}
    cache_path = cache_dir / "api_cache" / "fasta" / f"{species_id}.fa"
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    fasta_content = load_or_fetch_text(
        cache_path, metadata_path, url, params, refresh_cache
    )
    return sum(1 for line in fasta_content.splitlines() if line.startswith(">"))


def get_orthologs_between_species(
    species1_id: str,
    species2_id: str,
    clade: str | None,
    cache_dir: Path,
    refresh_cache: bool = False,
) -> list[list[dict[str, str]]]:
    """Load organism-human ortholog pairs from a cached OrthoDB API response."""
    url = f"https://data.orthodb.org/{ORTHODB_API_VERSION}/orthologs"
    params = {"species": species1_id, "species2": species2_id}
    if clade:
        params["clade"] = clade

    clade_label = clade or "none"
    cache_path = (
        cache_dir
        / "api_cache"
        / "orthologs"
        / f"{species1_id}__{species2_id}__clade_{clade_label}.json"
    )
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.json")
    content = load_or_fetch_text(cache_path, metadata_path, url, params, refresh_cache)
    data = json.loads(content)
    return data.get("data", [])


def parse_ortholog_pairs(
    ortholog_data: list[list[dict[str, str]]],
    organism_taxon_id: str,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """Parse OrthoDB API pairs and group all human orthologs per organism gene."""
    ortholog_pairs = []
    organism_to_human = defaultdict(list)

    for ortholog_pair in ortholog_data:
        if len(ortholog_pair) != 2:
            continue

        gene1, gene2 = ortholog_pair
        if gene1.get("param", "").startswith(organism_taxon_id):
            organism_gene = gene1
            human_gene = gene2
        else:
            organism_gene = gene2
            human_gene = gene1

        pair_info = {
            "organism_orthodb_id": organism_gene.get("param", ""),
            "organism_gene_name": organism_gene.get("id", ""),
            "human_orthodb_id": human_gene.get("param", ""),
            "human_gene_name": human_gene.get("id", ""),
        }

        ortholog_pairs.append(pair_info)
        organism_to_human[pair_info["organism_orthodb_id"]].append(pair_info)

    return ortholog_pairs, organism_to_human


def generate_csv_files(
    ortholog_pairs: list[dict[str, str]],
    organism_to_human: dict[str, list[dict[str, str]]],
    organism: str,
    output_dir: Path,
) -> None:
    """Generate all-pairs and bidirectional summary CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs_file = (
        output_dir / f"01_orthologs_{organism}_human_orthologs_all_pairs.csv"
    )
    with all_pairs_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "organism_orthodb_id",
            "organism_gene_name",
            "human_orthodb_id",
            "human_gene_name",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ortholog_pairs)

    organism_summary_file = (
        output_dir / f"01_orthologs_{organism}_human_orthologs_summary.csv"
    )
    with organism_summary_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "organism_orthodb_id",
            "organism_gene_name",
            "num_human_orthologs",
            "human_gene_names",
            "human_orthodb_ids",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for organism_id, human_orthologs in sorted(organism_to_human.items()):
            organism_name = human_orthologs[0]["organism_gene_name"]
            human_names = [pair["human_gene_name"] for pair in human_orthologs]
            human_ids = [pair["human_orthodb_id"] for pair in human_orthologs]
            writer.writerow(
                {
                    "organism_orthodb_id": organism_id,
                    "organism_gene_name": organism_name,
                    "num_human_orthologs": len(human_orthologs),
                    "human_gene_names": "; ".join(human_names),
                    "human_orthodb_ids": "; ".join(human_ids),
                }
            )

    human_to_organism = defaultdict(list)
    for pair in ortholog_pairs:
        human_to_organism[pair["human_orthodb_id"]].append(pair)

    human_summary_file = (
        output_dir / f"01_orthologs_human_{organism}_orthologs_summary.csv"
    )
    with human_summary_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "human_orthodb_id",
            "human_gene_name",
            "num_organism_orthologs",
            "organism_gene_names",
            "organism_orthodb_ids",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for human_id, organism_orthologs in sorted(human_to_organism.items()):
            human_name = organism_orthologs[0]["human_gene_name"]
            organism_names = [
                pair["organism_gene_name"] for pair in organism_orthologs
            ]
            organism_ids = [
                pair["organism_orthodb_id"] for pair in organism_orthologs
            ]
            writer.writerow(
                {
                    "human_orthodb_id": human_id,
                    "human_gene_name": human_name,
                    "num_organism_orthologs": len(organism_orthologs),
                    "organism_gene_names": "; ".join(organism_names),
                    "organism_orthodb_ids": "; ".join(organism_ids),
                }
            )


def print_statistics(
    ortholog_pairs: list[dict[str, str]],
    organism_to_human: dict[str, list[dict[str, str]]],
    total_organism_genes: int,
    organism_name: str,
) -> None:
    print("\n=== ORTHOLOG STATISTICS ===")
    print(f"Total ortholog pairs: {len(ortholog_pairs)}")
    print(f"Unique {organism_name} genes with human orthologs: {len(organism_to_human)}")
    if total_organism_genes:
        percent = len(organism_to_human) / total_organism_genes * 100
        print(f"Percentage of {organism_name} genes with human orthologs: {percent:.2f}%")
    unique_human_genes = {pair["human_orthodb_id"] for pair in ortholog_pairs}
    print(f"Unique human genes with {organism_name} orthologs: {len(unique_human_genes)}")
    multi_ortholog = sum(
        1 for orthologs in organism_to_human.values() if len(orthologs) > 1
    )
    one_to_one = sum(
        1 for orthologs in organism_to_human.values() if len(orthologs) == 1
    )
    print(f"{organism_name} genes with multiple human orthologs: {multi_ortholog}")
    print(f"One-to-one {organism_name} orthologs: {one_to_one}")


def write_run_manifest(
    organism: str,
    organism_taxon_id: str,
    human_id: str,
    clade: str | None,
    output_dir: Path,
    cache_dir: Path,
    total_organism_genes: int,
    ortholog_pairs: list[dict[str, str]],
    organism_to_human: dict[str, list[dict[str, str]]],
) -> None:
    manifest = {
        "orthodb_release": ORTHODB_RELEASE,
        "api_version": ORTHODB_API_VERSION,
        "organism": organism,
        "organism_orthodb_id": organism_taxon_id,
        "human_orthodb_id": human_id,
        "clade": clade,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_organism_genes": total_organism_genes,
        "total_ortholog_pairs": len(ortholog_pairs),
        "unique_organism_genes_with_human_orthologs": len(organism_to_human),
        "unique_human_genes": len(
            {pair["human_orthodb_id"] for pair in ortholog_pairs}
        ),
    }
    write_json(output_dir / f"01_orthologs_{organism}_human_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible organism-human ortholog CSV files."
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=sorted(ORGANISMS),
        help="Organism to process.",
    )
    parser.add_argument(
        "--clade",
        default=None,
        help="OrthoDB clade. Defaults to the organism-specific pinned value.",
    )
    parser.add_argument(
        "--human_id",
        default=HUMAN_ORTHODB_ID,
        help="Human OrthoDB species identifier.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ORTHODB_CACHE_DIR,
        help="Pinned OrthoDB cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for generated ortholog CSV files.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Download fresh OrthoDB API snapshots into the pinned cache.",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Write CSV outputs. Kept for compatibility with the old script.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    organism_config = ORGANISMS[args.organism]
    organism_taxon_id = organism_config["orthodb_id"]
    organism_name = organism_config["label"]
    clade = args.clade or organism_config["default_clade"]

    print(f"=== Generating {organism_name}-human ortholog CSVs ===")
    print(f"OrthoDB release: {ORTHODB_RELEASE} ({ORTHODB_API_VERSION})")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Refresh cache: {args.refresh_cache}")

    total_organism_genes = get_all_genes_in_species(
        organism_taxon_id,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    print(f"Total {organism_name} genes: {total_organism_genes}")

    ortholog_data = get_orthologs_between_species(
        args.human_id,
        organism_taxon_id,
        clade=clade,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    if not ortholog_data:
        print("No ortholog data available. Exiting.", file=sys.stderr)
        return 1

    print(f"Retrieved {len(ortholog_data)} ortholog pairs")
    ortholog_pairs, organism_to_human = parse_ortholog_pairs(
        ortholog_data,
        organism_taxon_id=organism_taxon_id,
    )

    if args.save_data:
        generate_csv_files(
            ortholog_pairs,
            organism_to_human,
            organism=args.organism,
            output_dir=args.output_dir,
        )
        write_run_manifest(
            organism=args.organism,
            organism_taxon_id=organism_taxon_id,
            human_id=args.human_id,
            clade=clade,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            total_organism_genes=total_organism_genes,
            ortholog_pairs=ortholog_pairs,
            organism_to_human=organism_to_human,
        )
        print(f"Saved outputs to: {args.output_dir}")
    else:
        print("Skipping CSV generation. Pass --save_data to write outputs.")

    print_statistics(
        ortholog_pairs,
        organism_to_human,
        total_organism_genes,
        organism_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
