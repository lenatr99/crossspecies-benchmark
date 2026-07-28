#!/usr/bin/env python3
"""Fit ESM2 calibration adapters for UCE protein tokens.

By default, this anchor-free calibration rescales a source-species ESM2 protein
embedding cloud so each dimension matches the human ESM2 mean and standard
deviation. With ``--tiers tier1``, the whitening statistics are estimated from
matched Tier 1 ortholog pairs. In both modes all source genes are transformed
with one global rule; no source gene is pulled toward a specific human gene.

The ``macrogene`` method keeps global whitening as the species-wide OOD
correction, then applies a small Tier 1 residual correction through soft
protein-archetype assignments. This follows SATURN's macrogene intuition
without training a cell-level model: orthologs guide conserved neighborhoods,
but the broad protein geometry remains globally calibrated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection

from embed import (
    DATASETS,
    DEFAULT_UCE_DIR,
    HUMAN_PROTEIN_EMBEDDING,
    environment_metadata,
    ensure_drive_protein_embedding,
    ensure_protein_embedding_assets,
    file_manifest,
    files_manifest,
    git_repro_metadata,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_DIR = ROOT / "models" / "UCE" / "model_files" / "protein_embeddings"
DEFAULT_ORTHOLOG_DIR = ROOT / "results" / "orthologs"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "embeddings" / "uce" / "esm2_adapter"

HUMAN_EMBEDDING = HUMAN_PROTEIN_EMBEDDING
SPECIES_EMBEDDINGS = {
    "dicty": "Dictyostelium_discoideum.dicty_2.7.pep.all.gene_symbol_to_embedding_ESM2.pt",
    "elegans": "Caenorhabditis_elegans.WBcel235.pep.all.gene_symbol_to_embedding_ESM2.pt",
}

EPS = 1e-8
METHOD_CHOICES = ("whitening", "macrogene")
TIER_CHOICES = ("global", "tier1")
MACROGENE_TIERS = "global"


def load_embedding_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing protein embedding: {path}. Run through "
            "code/embeddings/uce/run_uce_locally.sh --whitening so UCE assets "
            "are downloaded before fitting, or place the file under "
            "models/UCE/model_files/protein_embeddings/."
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(payload)!r}")
    return payload


def stack_embeddings(embeddings: dict[str, torch.Tensor], genes: list[str]) -> np.ndarray:
    rows = []
    for gene in genes:
        value = embeddings[gene]
        arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        rows.append(np.asarray(arr, dtype=np.float32))
    return np.vstack(rows)


def as_torch_dict(genes: list[str], matrix: np.ndarray) -> dict[str, torch.Tensor]:
    return {gene: torch.as_tensor(matrix[i], dtype=torch.float32) for i, gene in enumerate(genes)}


def fit_whitening(x: np.ndarray, y: np.ndarray, *, method: str) -> dict[str, np.ndarray | str]:
    return {
        "method": method,
        "x_mean": x.mean(axis=0).astype(np.float64),
        "x_std": x.std(axis=0).astype(np.float64),
        "y_mean": y.mean(axis=0).astype(np.float64),
        "y_std": y.std(axis=0).astype(np.float64),
    }


def apply_whitening(matrix: np.ndarray, fit: dict[str, np.ndarray | str]) -> np.ndarray:
    x = matrix.astype(np.float64)
    z = (x - fit["x_mean"]) / (fit["x_std"] + EPS)
    return (z * fit["y_std"] + fit["y_mean"]).astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize(a, norm="l2")
    b_n = normalize(b, norm="l2")
    return np.sum(a_n * b_n, axis=1)


def nearest_cosine(query: np.ndarray, reference: np.ndarray, batch_size: int) -> np.ndarray:
    q = normalize(query.astype(np.float32), norm="l2")
    r = normalize(reference.astype(np.float32), norm="l2")
    out = np.empty(q.shape[0], dtype=np.float32)
    for i in range(0, q.shape[0], batch_size):
        sims = q[i : i + batch_size] @ r.T
        out[i : i + batch_size] = sims.max(axis=1)
    return out


def softmax_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled.astype(np.float64))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def soft_assignments(
    matrix: np.ndarray,
    centers: np.ndarray,
    *,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    x_n = normalize(matrix.astype(np.float32), norm="l2")
    c_n = normalize(centers.astype(np.float32), norm="l2")
    out = np.empty((matrix.shape[0], centers.shape[0]), dtype=np.float32)
    for i in range(0, matrix.shape[0], batch_size):
        sims = x_n[i : i + batch_size] @ c_n.T
        out[i : i + batch_size] = softmax_rows(sims, temperature)
    return out


def normalized_projection(
    *,
    source: np.ndarray,
    human: np.ndarray,
    projection_dim: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, GaussianRandomProjection | None]:
    source_n = normalize(source.astype(np.float32), norm="l2")
    human_n = normalize(human.astype(np.float32), norm="l2")
    if projection_dim <= 0 or projection_dim >= source_n.shape[1]:
        return source_n, human_n, None

    projector = GaussianRandomProjection(
        n_components=projection_dim,
        random_state=random_state,
    )
    train = np.vstack([source_n, human_n])
    train_proj = projector.fit_transform(train).astype(np.float32)
    return (
        train_proj[: source_n.shape[0]],
        train_proj[source_n.shape[0] :],
        projector,
    )


def preserve_row_norms(adapted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    adapted_norm = np.linalg.norm(adapted, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    return (adapted / (adapted_norm + EPS) * reference_norm).astype(np.float32)


def fit_macrogene_adapter(
    *,
    x_global: np.ndarray,
    y: np.ndarray,
    anchor_source: np.ndarray,
    anchor_human: np.ndarray,
    n_macrogenes: int,
    beta: float,
    temperature: float,
    projection_dim: int,
    random_state: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    if anchor_source.shape[0] == 0:
        raise ValueError("Macrogene adapter requires Tier 1 anchor pairs")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("--macrogene-beta must be between 0 and 1")
    if temperature <= 0:
        raise ValueError("--macrogene-temperature must be positive")

    source_features, human_features, projector = normalized_projection(
        source=x_global,
        human=y,
        projection_dim=projection_dim,
        random_state=random_state,
    )
    train = np.vstack([source_features, human_features])
    kmeans = MiniBatchKMeans(
        n_clusters=n_macrogenes,
        batch_size=max(1024, batch_size),
        max_iter=100,
        n_init=3,
        random_state=random_state,
        reassignment_ratio=0.01,
    )
    kmeans.fit(train)
    centers = kmeans.cluster_centers_.astype(np.float32)

    q_source = soft_assignments(
        source_features,
        centers,
        temperature=temperature,
        batch_size=batch_size,
    )
    if projector is None:
        anchor_features = normalize(anchor_source.astype(np.float32), norm="l2")
    else:
        anchor_features = projector.transform(
            normalize(anchor_source.astype(np.float32), norm="l2")
        ).astype(np.float32)
    q_anchor = soft_assignments(
        anchor_features,
        centers,
        temperature=temperature,
        batch_size=batch_size,
    )
    anchor_residual = (anchor_human - anchor_source).astype(np.float32)
    weights = q_anchor.sum(axis=0).astype(np.float64)
    residual_by_macrogene = (q_anchor.T.astype(np.float64) @ anchor_residual.astype(np.float64))
    residual_by_macrogene = (
        residual_by_macrogene / (weights[:, None] + EPS)
    ).astype(np.float32)

    residual = q_source @ residual_by_macrogene
    x_after = preserve_row_norms(x_global + beta * residual, x_global)

    center_diag = pd.DataFrame(
        {
            "macrogene": np.arange(n_macrogenes, dtype=int),
            "anchor_weight": weights,
            "residual_norm": np.linalg.norm(residual_by_macrogene, axis=1),
        }
    )
    metadata: dict[str, object] = {
        "n_macrogenes": int(n_macrogenes),
        "beta": float(beta),
        "temperature": float(temperature),
        "projection_dim": int(projection_dim),
        "projected_feature_dim": int(train.shape[1]),
        "random_state": int(random_state),
        "n_anchor_pairs": int(anchor_source.shape[0]),
        "mean_anchor_assignment_entropy": float(
            -(q_anchor * np.log(q_anchor + EPS)).sum(axis=1).mean()
        ),
        "mean_source_assignment_entropy": float(
            -(q_source * np.log(q_source + EPS)).sum(axis=1).mean()
        ),
        "mean_macrogene_residual_norm": float(
            np.linalg.norm(residual_by_macrogene, axis=1).mean()
        ),
    }
    return x_after, metadata, center_diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SPECIES_EMBEDDINGS), required=True)
    parser.add_argument("--uce-dir", type=Path, default=DEFAULT_UCE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="whitening",
        help=(
            "'whitening' writes the original global/Tier 1 moment adapter. "
            "'macrogene' applies global whitening plus a small Tier 1 "
            "protein-archetype residual correction."
        ),
    )
    parser.add_argument(
        "--tiers",
        choices=TIER_CHOICES,
        default="global",
        help=(
            "Genes used to fit whitening statistics. 'global' uses all source "
            "and all human ESM2 embeddings; 'tier1' uses matched Tier 1 "
            "ortholog pairs, then applies the fitted transform to all source genes."
        ),
    )
    parser.add_argument("--n-macrogenes", type=int, default=128)
    parser.add_argument("--macrogene-beta", type=float, default=0.1)
    parser.add_argument("--macrogene-temperature", type=float, default=0.07)
    parser.add_argument(
        "--macrogene-projection-dim",
        type=int,
        default=256,
        help="Random-projection dimension for protein archetype clustering; use 0 to disable.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--diag-batch-size", type=int, default=500)
    return parser.parse_args()


def tier_fit_matrices(
    *,
    dataset: str,
    tiers: str,
    source_emb: dict[str, torch.Tensor],
    human_emb: dict[str, torch.Tensor],
    source_genes: list[str],
    human_genes: list[str],
    source_matrix: np.ndarray,
    human_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, Path | None]:
    if tiers == "global":
        return source_matrix, human_matrix, int(source_matrix.shape[0]), None

    pair_path = DEFAULT_ORTHOLOG_DIR / f"02_consensus_{dataset}.tsv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Missing ortholog consensus table for {tiers} whitening: {pair_path}")

    tier_num = int(tiers.removeprefix("tier"))
    pairs = pd.read_csv(pair_path, sep="\t")
    pairs = pairs[pairs["tier"].astype(int) == tier_num].copy()
    pairs = pairs.sort_values("n_sources", ascending=False).drop_duplicates("organism_gene", keep="first")
    pairs = pairs[
        pairs["organism_gene"].astype(str).isin(source_emb)
        & pairs["human_symbol"].astype(str).isin(human_emb)
    ].copy()
    if pairs.empty:
        raise ValueError(f"No usable {tiers} ortholog pairs found in {pair_path}")

    src_idx = {g: i for i, g in enumerate(source_genes)}
    hum_idx = {g: i for i, g in enumerate(human_genes)}
    source_indices = [src_idx[str(g)] for g in pairs["organism_gene"]]
    human_indices = [hum_idx[str(g)] for g in pairs["human_symbol"]]
    return source_matrix[source_indices], human_matrix[human_indices], int(len(pairs)), pair_path


def main() -> int:
    args = parse_args()
    if args.method == "macrogene" and args.tiers != MACROGENE_TIERS:
        raise ValueError("The macrogene adapter is global-first; use --tiers global")

    run_name = f"{args.dataset}_{args.method}_{args.tiers}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.embedding_dir / SPECIES_EMBEDDINGS[args.dataset]
    human_path = args.embedding_dir / HUMAN_EMBEDDING
    if args.embedding_dir == DEFAULT_EMBEDDING_DIR:
        source_path = ensure_drive_protein_embedding(args.uce_dir, DATASETS[args.dataset].protein_embedding_name)
        human_path = args.uce_dir / "model_files" / "protein_embeddings" / HUMAN_EMBEDDING
        if not human_path.exists() or human_path.stat().st_size == 0:
            ensure_protein_embedding_assets(args.uce_dir)

    source_emb = load_embedding_dict(source_path)
    human_emb = load_embedding_dict(human_path)
    source_genes = list(source_emb.keys())
    human_genes = list(human_emb.keys())
    x = stack_embeddings(source_emb, source_genes)
    y = stack_embeddings(human_emb, human_genes)

    fit_x, fit_y, n_fit_genes, fit_pair_path = tier_fit_matrices(
        dataset=args.dataset,
        tiers=args.tiers,
        source_emb=source_emb,
        human_emb=human_emb,
        source_genes=source_genes,
        human_genes=human_genes,
        source_matrix=x,
        human_matrix=y,
    )
    fit = fit_whitening(fit_x, fit_y, method=args.method)
    x_global = apply_whitening(x, fit)
    x_after = x_global
    method_metadata: dict[str, object] = {}
    center_diag_path: Path | None = None

    if args.method == "macrogene":
        anchor_x, anchor_y, n_anchor_genes, anchor_pair_path = tier_fit_matrices(
            dataset=args.dataset,
            tiers="tier1",
            source_emb=source_emb,
            human_emb=human_emb,
            source_genes=source_genes,
            human_genes=human_genes,
            source_matrix=x_global,
            human_matrix=y,
        )
        x_after, method_metadata, center_diag = fit_macrogene_adapter(
            x_global=x_global,
            y=y,
            anchor_source=anchor_x,
            anchor_human=anchor_y,
            n_macrogenes=args.n_macrogenes,
            beta=args.macrogene_beta,
            temperature=args.macrogene_temperature,
            projection_dim=args.macrogene_projection_dim,
            random_state=args.random_state,
            batch_size=args.diag_batch_size,
        )
        n_fit_genes = n_anchor_genes
        fit_pair_path = anchor_pair_path
        center_diag_path = output_dir / f"{run_name}_macrogenes.csv"
        center_diag.to_csv(center_diag_path, index=False)

    nn_before = nearest_cosine(x, y, args.diag_batch_size)
    nn_after = nearest_cosine(x_after, y, args.diag_batch_size)

    diag = pd.DataFrame(
        {
            "gene": source_genes,
            "nn_cosine_before": nn_before,
            "nn_cosine_after": nn_after,
            "nn_cosine_delta": nn_after - nn_before,
        }
    )

    pair_diag: pd.DataFrame | None = None
    pair_path = DEFAULT_ORTHOLOG_DIR / f"02_consensus_{args.dataset}.tsv"
    if pair_path.exists():
        pairs = pd.read_csv(pair_path, sep="\t")
        pairs = pairs.sort_values("tier").drop_duplicates("organism_gene", keep="first")
        pairs = pairs[
            pairs["organism_gene"].astype(str).isin(source_emb)
            & pairs["human_symbol"].astype(str).isin(human_emb)
        ].copy()
        if not pairs.empty:
            src_idx = {g: i for i, g in enumerate(source_genes)}
            hum_idx = {g: i for i, g in enumerate(human_genes)}
            si = [src_idx[str(g)] for g in pairs["organism_gene"]]
            hi = [hum_idx[str(g)] for g in pairs["human_symbol"]]
            assigned_before = cosine_rows(x[si], y[hi])
            assigned_after = cosine_rows(x_after[si], y[hi])
            pair_diag = pairs[["organism_gene", "human_symbol", "tier"]].copy()
            pair_diag["assigned_cosine_before"] = assigned_before
            pair_diag["assigned_cosine_after"] = assigned_after
            pair_diag["assigned_cosine_delta"] = assigned_after - assigned_before

    diag_path = output_dir / f"{run_name}_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    pair_diag_path: Path | None = None
    if pair_diag is not None:
        pair_diag_path = output_dir / f"{run_name}_pair_diagnostics.csv"
        pair_diag.to_csv(pair_diag_path, index=False)

    adapted_path = output_dir / f"{run_name}_gene_symbol_to_embedding_ESM2_adapted.pt"
    torch.save(as_torch_dict(source_genes, x_after), adapted_path)

    metrics: dict[str, float] = {
        "mean_nn_cosine_before": float(nn_before.mean()),
        "mean_nn_cosine_after": float(nn_after.mean()),
        "median_nn_cosine_before": float(np.median(nn_before)),
        "median_nn_cosine_after": float(np.median(nn_after)),
        "source_per_dim_mean_norm_before": float(np.linalg.norm(x.mean(axis=0))),
        "source_per_dim_mean_norm_after": float(np.linalg.norm(x_after.mean(axis=0))),
        "global_whitened_per_dim_mean_norm": float(np.linalg.norm(x_global.mean(axis=0))),
        "human_per_dim_mean_norm": float(np.linalg.norm(y.mean(axis=0))),
        "source_per_dim_std_mean_before": float(x.std(axis=0).mean()),
        "source_per_dim_std_mean_after": float(x_after.std(axis=0).mean()),
        "global_whitened_per_dim_std_mean": float(x_global.std(axis=0).mean()),
        "human_per_dim_std_mean": float(y.std(axis=0).mean()),
    }
    if pair_diag is not None:
        for tier_val, sub in pair_diag.groupby("tier"):
            metrics[f"tier{int(tier_val)}_assigned_cosine_before"] = float(
                sub["assigned_cosine_before"].mean()
            )
            metrics[f"tier{int(tier_val)}_assigned_cosine_after"] = float(
                sub["assigned_cosine_after"].mean()
            )

    manifest = {
        "step": "fit_esm2_adapter",
        "dataset": args.dataset,
        "method": args.method,
        "method_slug": args.method,
        "tag": args.tiers,
        "tiers_label": args.tiers,
        "method_parameters": method_metadata,
        "source_embedding": str(source_path),
        "human_embedding": str(human_path),
        "n_source_genes": int(len(source_genes)),
        "n_human_genes": int(len(human_genes)),
        "n_fit_genes": n_fit_genes,
        "fit_pair_table": str(fit_pair_path) if fit_pair_path is not None else None,
        "adapted_embedding": str(adapted_path),
        "diagnostics": str(diag_path),
        "pair_diagnostics": str(pair_diag_path) if pair_diag_path else None,
        "macrogene_diagnostics": str(center_diag_path) if center_diag_path else None,
        "metrics": metrics,
        "reproducibility": {
            "environment": environment_metadata(),
            "uce_git": git_repro_metadata(args.uce_dir),
            "asset_files": files_manifest(
                [
                    source_path,
                    human_path,
                    adapted_path,
                    diag_path,
                    *( [pair_diag_path] if pair_diag_path is not None else [] ),
                    *( [center_diag_path] if center_diag_path is not None else [] ),
                    *( [fit_pair_path] if fit_pair_path is not None else [] ),
                    pair_path,
                ]
            ),
            "source_embedding_file": file_manifest(source_path),
            "human_embedding_file": file_manifest(human_path),
            "adapted_embedding_file": file_manifest(adapted_path),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / f"{run_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote adapted embeddings: {adapted_path}")
    print(f"Wrote diagnostics: {diag_path}")
    if pair_diag_path is not None:
        print(f"Wrote pair diagnostics: {pair_diag_path}")
    if center_diag_path is not None:
        print(f"Wrote macrogene diagnostics: {center_diag_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(
        f"Mean nearest-human cosine: {nn_before.mean():.4f} -> {nn_after.mean():.4f}"
        f"  (median {np.median(nn_before):.4f} -> {np.median(nn_after):.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
