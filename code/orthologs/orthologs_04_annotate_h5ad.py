"""Step 04: add consensus human ortholog metadata to AnnData gene tables.

Orthologs are gene-level annotations, so they belong in ``adata.var`` rather
than ``adata.obs``. This script updates only the H5AD ``/var`` table in place,
leaving the expression matrix and cell metadata untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import pandas as pd

try:
    from anndata.io import write_elem
except ImportError:  # pragma: no cover - older anndata fallback
    from anndata.experimental import write_elem


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "results" / "orthologs" / "04_h5ad_ortholog_annotation_manifest.json"

DATASETS = {
    "dicty": {
        "h5ad": ROOT / "data" / "dicty.h5ad",
        "consensus": ROOT / "results" / "orthologs" / "02_consensus_dicty.tsv",
        "mapping": ROOT / "data" / "ortholog_mapping_orthodb.json",
        "gene_id_col": "gene_ids",
    },
    "elegans": {
        "h5ad": ROOT / "data" / "elegans.h5ad",
        "consensus": ROOT / "results" / "orthologs" / "02_consensus_elegans.tsv",
        "mapping": ROOT / "data" / "ortholog_mapping_elegans_orthodb.json",
        "gene_id_col": "id",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for gene_id, symbols in raw.items():
        if isinstance(symbols, list) and symbols:
            out[str(gene_id)] = str(symbols[0])
    return out


def build_consensus_lookup(consensus_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(consensus_tsv, sep="\t", dtype=str).fillna("")
    keep_cols = [
        "organism_gene",
        "human_symbol",
        "tier",
        "n_sources",
        "sources",
        "orthofinder_type",
        "orthodb_og_category",
        "oma_rel_type",
        "curated",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{consensus_tsv} missing required columns: {missing}")
    return df[keep_cols].drop_duplicates("organism_gene").set_index("organism_gene")


def annotate_var(
    var: pd.DataFrame,
    *,
    gene_id_col: str,
    final_mapping: dict[str, str],
    consensus: pd.DataFrame,
) -> pd.DataFrame:
    if gene_id_col not in var.columns:
        raise ValueError(f"Expected adata.var[{gene_id_col!r}], found {list(var.columns)}")

    out = var.copy()
    gene_ids = out[gene_id_col].astype(str)

    final_symbol = gene_ids.map(final_mapping).fillna("No_ortholog")
    consensus_rows = consensus.reindex(gene_ids)

    old_columns = [
        "ortholog",
        "ortholog_gene_id",
        "human_ortholog_symbol",
        "has_human_ortholog",
        "ortholog_retained_tier_le_2",
        "ortholog_best_human_symbol_any_tier",
        "ortholog_tier",
        "ortholog_n_sources",
        "ortholog_sources",
        "ortholog_orthofinder_type",
        "ortholog_orthodb_og_category",
        "ortholog_oma_rel_type",
        "ortholog_curated_sources",
    ]
    out = out.drop(columns=[c for c in old_columns if c in out.columns])

    best_any_tier = consensus_rows["human_symbol"].fillna("")
    tier = consensus_rows["tier"].fillna("")
    for tier_id in ("1", "2", "3"):
        out[f"ortholog_tier_{tier_id}"] = best_any_tier.where(tier == tier_id, "").to_numpy()
    return out


def update_h5ad_var(path: Path, var: pd.DataFrame) -> None:
    with h5py.File(path, "a") as handle:
        if "var" in handle:
            del handle["var"]
        write_elem(handle, "var", var)


def annotate_dataset(name: str) -> dict[str, object]:
    cfg = DATASETS[name]
    h5ad_path = Path(cfg["h5ad"])
    mapping_path = Path(cfg["mapping"])
    consensus_path = Path(cfg["consensus"])
    gene_id_col = str(cfg["gene_id_col"])

    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    if not consensus_path.exists():
        raise FileNotFoundError(consensus_path)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        var = adata.var.copy()
        shape = adata.shape
    finally:
        adata.file.close()

    final_mapping = load_mapping(mapping_path)
    consensus = build_consensus_lookup(consensus_path)
    annotated = annotate_var(
        var,
        gene_id_col=gene_id_col,
        final_mapping=final_mapping,
        consensus=consensus,
    )
    update_h5ad_var(h5ad_path, annotated)

    n_tier_1 = int(annotated["ortholog_tier_1"].astype(str).ne("").sum())
    n_tier_2 = int(annotated["ortholog_tier_2"].astype(str).ne("").sum())
    n_tier_3 = int(annotated["ortholog_tier_3"].astype(str).ne("").sum())
    return {
        "dataset": name,
        "h5ad": str(h5ad_path),
        "shape": tuple(int(x) for x in shape),
        "gene_id_col": gene_id_col,
        "n_genes": int(annotated.shape[0]),
        "n_ortholog_tier_1": n_tier_1,
        "n_ortholog_tier_2": n_tier_2,
        "n_ortholog_tier_3": n_tier_3,
        "mapping_json": str(mapping_path),
        "mapping_json_sha256": sha256_file(mapping_path),
        "consensus_tsv": str(consensus_path),
        "consensus_tsv_sha256": sha256_file(consensus_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate dicty/elegans AnnData var tables with consensus orthologs."
    )
    parser.add_argument(
        "--dataset",
        choices=[*sorted(DATASETS), "all"],
        default="all",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Where to write the H5AD annotation manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset in datasets:
        summary = annotate_dataset(dataset)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    manifest = {
        "step": "04_annotate_h5ad",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Ortholog annotations are gene-level metadata written to adata.var, not cell-level adata.obs.",
        "datasets": summaries,
        "columns_added_or_refreshed": [
            "ortholog_tier_1",
            "ortholog_tier_2",
            "ortholog_tier_3",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
