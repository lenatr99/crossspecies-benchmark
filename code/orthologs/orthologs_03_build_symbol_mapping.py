"""Step 03: emit final ``{organism_gene: [human_symbol]}`` JSON + evidence sidecar.

Replaces the legacy ``orthologs_04_build_symbol_mapping.py`` while keeping the
output path and shape byte-compatible so downstream notebooks
(``embed_geneformer.ipynb``, ``embed_scGPT.ipynb``, ``embed_UCE.ipynb`` and
their elegans variants) keep working unchanged.

Reads ``02_consensus_<organism>.tsv`` and:

  * Emits ``data/ortholog_mapping_<organism>_orthodb.json`` containing
    ``{organism_gene: [human_symbol]}`` filtered to ``--min-tier`` (default 2,
    so only Tier 1 + Tier 2 land in the JSON consumed by foundation models).
  * Emits a sidecar ``data/ortholog_mapping_<organism>_orthodb.evidence.json``
    with full per-gene evidence for ablation/figures.
  * Emits a manifest with sha256 of every input + tool versions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _utils import sha256_file, utc_now, write_json
from config import (
    EVIDENCE_SOURCES,
    ORGANISMS,
    ORTHODB_RELEASE,
    RESULTS_DIR,
)


def filter_min_tier(df: pd.DataFrame, min_tier: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["tier"].astype(int) <= min_tier].copy()


def build_main_mapping(df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for organism_gene, sub in df.groupby("organism_gene"):
        first = sub.iloc[0]
        out[organism_gene] = [first["human_symbol"]]
    return out


def build_evidence_sidecar(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        organism_gene = row["organism_gene"]
        out[organism_gene] = {
            "human_symbol": row["human_symbol"],
            "tier": int(row["tier"]),
            "n_sources": int(row["n_sources"]),
            "sources": str(row["sources"] or "").split(";") if row["sources"] else [],
            "orthofinder_type": row.get("orthofinder_type", "") or "",
            "orthofinder_orthogroup": row.get("orthofinder_orthogroup", "") or "",
            "orthodb_og_category": row.get("orthodb_og_category", "") or "",
            "orthodb_og_id": row.get("orthodb_og_id", "") or "",
            "oma_rel_type": row.get("oma_rel_type", "") or "",
            "curated": str(row.get("curated", "") or "").split(";") if row.get("curated") else [],
            "all_candidates": json.loads(row["all_candidates_json"]) if row.get("all_candidates_json") else {},
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final {organism_gene: [human_symbol]} JSON used by foundation models."
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=sorted(ORGANISMS),
    )
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--min-tier",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Lowest tier number to retain (1=gold, 2=+silver, 3=+bronze). Default: 2.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ORGANISMS[args.organism]
    consensus_tsv = args.input_dir / f"02_consensus_{args.organism}.tsv"
    output = Path(args.output) if args.output else cfg["mapping_output"]
    sidecar = output.with_suffix("")
    sidecar = sidecar.parent / (sidecar.name + ".evidence.json")

    if not consensus_tsv.exists():
        raise FileNotFoundError(
            f"Missing consensus TSV at {consensus_tsv}. Run orthologs_02_consensus.py first."
        )

    df = pd.read_csv(consensus_tsv, sep="\t")
    df_main = filter_min_tier(df, args.min_tier)
    df_main = df_main.sort_values(["tier", "organism_gene"]).reset_index(drop=True)
    mapping = build_main_mapping(df_main)
    evidence = build_evidence_sidecar(df)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, mapping)
    write_json(sidecar, evidence)

    tier_counts = (
        df_main["tier"].value_counts().sort_index().to_dict() if not df_main.empty else {}
    )

    manifest = {
        "step": "03_build_symbol_mapping",
        "orthodb_release": ORTHODB_RELEASE,
        "organism": args.organism,
        "evidence_sources": EVIDENCE_SOURCES,
        "consensus_tsv": str(consensus_tsv),
        "consensus_sha256": sha256_file(consensus_tsv),
        "min_tier": args.min_tier,
        "output_json": str(output),
        "output_evidence_json": str(sidecar),
        "n_genes_in_main": len(mapping),
        "n_genes_with_evidence": len(evidence),
        "tier_counts_in_main": {str(k): int(v) for k, v in tier_counts.items()},
        "generated_utc": utc_now(),
    }
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)

    print(
        f"[{args.organism}] tier ≤ {args.min_tier}: kept {len(mapping)} of {len(df)} candidates "
        f"({tier_counts})"
    )
    print(f"Wrote main JSON  : {output}")
    print(f"Wrote evidence   : {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
