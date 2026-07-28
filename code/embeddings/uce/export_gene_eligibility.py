"""Export the exact chromosome-aware UCE gene eligibility sets.

This publication helper applies the same case-insensitive conjunction used by
``embed.py``: a gene must occur in both the official species protein dictionary
and the generated chromosome-position asset.  It does not create embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UCE_DIR = ROOT / "models" / "UCE"
OUT_DIR = ROOT / "results" / "supplements"

ASSETS = {
    "dicty": {
        "species": "dicty",
        "protein": UCE_DIR / "model_files" / "protein_embeddings"
        / "Dictyostelium_discoideum.dicty_2.7.pep.all.gene_symbol_to_embedding_ESM2.pt",
        "chromosome": UCE_DIR / "data_proc" / "dicty_to_chrom_pos.csv",
    },
    "elegans": {
        "species": "elegans",
        "protein": UCE_DIR / "model_files" / "protein_embeddings"
        / "Caenorhabditis_elegans.WBcel235.pep.all.gene_symbol_to_embedding_ESM2.pt",
        "chromosome": UCE_DIR / "data_proc" / "elegans_to_chrom_pos.csv",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> int:
    import pandas as pd
    import torch

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    tsv_rows = []
    for organism, cfg in ASSETS.items():
        protein_path = cfg["protein"]
        chrom_path = cfg["chromosome"]
        if not protein_path.is_file() or not chrom_path.is_file():
            raise FileNotFoundError(f"Missing UCE assets for {organism}")
        protein = torch.load(protein_path, map_location="cpu")
        protein_genes = {str(key).upper() for key in protein}
        chrom = pd.read_csv(chrom_path)
        chrom_genes = set(
            chrom.loc[
                chrom["species"].astype(str) == cfg["species"], "gene_symbol"
            ].astype(str).str.upper()
        )
        eligible = sorted(protein_genes.intersection(chrom_genes))
        records[organism] = {
            "species_token": cfg["species"],
            "protein_path": str(protein_path),
            "protein_sha256": sha256_file(protein_path),
            "chromosome_path": str(chrom_path),
            "chromosome_sha256": sha256_file(chrom_path),
            "n_protein_dictionary_genes": len(protein_genes),
            "n_chromosome_genes": len(chrom_genes),
            "n_exact_chromosome_aware_eligible_genes": len(eligible),
            "eligible_gene_ids": eligible,
        }
        tsv_rows.extend((organism, gene) for gene in eligible)
        del protein

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility_rule": (
            "uppercase(gene) in official protein dictionary AND uppercase(gene) "
            "in species chromosome-position asset"
        ),
        "chromosome_aware": True,
        "organisms": records,
    }
    json_path = args.output_dir / "uce_gene_eligibility.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv_path = args.output_dir / "uce_gene_eligibility.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["organism", "gene_identifier"])
        writer.writerows(tsv_rows)
    print(f"wrote {json_path}")
    print(f"wrote {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
