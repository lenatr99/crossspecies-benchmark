"""Compute and save UMAP visualizations for foundation-model embeddings.

This shared postprocessing script looks for embedding matrices in
``adata.obsm``, computes 2D UMAP coordinates, writes those coordinates back to
the source H5AD, and saves Scanpy-style colored panels under
``results/embeddings/umaps``.

Use ``--model`` for common embeddings (geneformer, pca, scvi, scgpt, uce).
Pass ``--embedding-template`` or explicit ``--embedding-keys`` for odd cases.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from plot_umap_presets import MODEL_PRESETS, REPRESENTATION_SUFFIXES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results" / "embeddings" / "umaps"
DEFAULT_EMBEDDING_TEMPLATE = "X_geneformer_ortholog_{variant}{representation_suffix}"

# Keep Scanpy/Matplotlib/Numba caches out of user homes and read-only package
# locations. This avoids cache errors in fresh envs and makes repeated plotting
# runs less fussy.
DEFAULT_CACHE_DIR = ROOT / "data" / "foundation_models" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))


DATASETS = {
    "dicty": {
        "path": ROOT / "data" / "dicty.h5ad",
        "label": "dicty",
        "colors": ("time", "cell_type"),
    },
    "elegans": {
        "path": ROOT / "data" / "elegans.h5ad",
        "label": "elegans",
        "colors": ("time", "cell_type"),
    },
}

def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def format_embedding_key(template: str, variant: str, representation: str) -> str:
    suffix = REPRESENTATION_SUFFIXES[representation]
    return template.format(
        variant=variant,
        representation=representation,
        representation_suffix=suffix,
    )


def umap_key_for(embedding_key: str) -> str:
    return f"{embedding_key}_umap"


def infer_layer(embedding_key: str, model: str) -> str:
    if "4layer" in embedding_key or "_4L" in embedding_key:
        return "4layer"
    if "33layer" in embedding_key or "_33L" in embedding_key:
        return "33layer"
    return model


def output_dir_for(
    results_dir: Path,
    template: str,
    *,
    dataset: str,
    model: str,
    output_prefix: str,
    item: dict[str, object],
) -> Path:
    embedding_key = str(item["embedding_key"])
    safe_key = embedding_key.removeprefix("X_").replace("/", "_").replace(" ", "_")
    fields = {
        "dataset": dataset,
        "model": model,
        "output_prefix": output_prefix,
        "variant": item.get("variant") or "all",
        "representation": item.get("representation") or "all",
        "embedding_key": safe_key,
        "layer": infer_layer(embedding_key, model),
    }
    return results_dir / Path(template.format(**fields))


def write_obsm_array(h5ad_path: Path, key: str, array) -> None:
    import h5py
    import numpy as np

    try:
        from anndata.io import write_elem
    except ImportError:  # pragma: no cover - older anndata fallback
        from anndata.experimental import write_elem

    with h5py.File(h5ad_path, "a") as handle:
        if "obsm" not in handle:
            handle.create_group("obsm")
        obsm = handle["obsm"]
        if key in obsm:
            del obsm[key]
        write_elem(obsm, key, np.asarray(array))


def compute_umap(adata, embedding_key: str, *, random_state: int, min_dist: float, n_neighbors: int):
    import umap

    reducer = umap.UMAP(
        random_state=random_state,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
    )
    return reducer.fit_transform(adata.obsm[embedding_key])


def save_embedding_plot(
    adata,
    *,
    basis: str,
    color: str,
    output_path: Path,
    size: float,
    alpha: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        size=size,
        alpha=alpha,
        frameon=False,
        show=False,
        return_fig=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_dataset(
    *,
    dataset: str,
    model: str,
    variants: list[str],
    representations: list[str],
    embedding_template: str,
    embedding_keys: list[str] | None,
    output_prefix: str,
    output_dir_template: str,
    colors: list[str] | None,
    results_dir: Path,
    random_state: int,
    min_dist: float,
    n_neighbors: int,
    size: float,
    alpha: float,
    dpi: int,
    overwrite: bool,
    skip_missing: bool,
) -> dict[str, object]:
    import anndata as ad

    cfg = DATASETS[dataset]
    h5ad_path = cfg["path"]
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    adata = ad.read_h5ad(h5ad_path)
    plot_colors = colors if colors is not None else list(cfg["colors"])
    missing_colors = [c for c in plot_colors if c not in adata.obs.columns]
    if missing_colors:
        raise ValueError(f"{h5ad_path} missing obs color column(s): {missing_colors}")

    outputs: list[dict[str, object]] = []
    output_dirs: set[Path] = set()
    if embedding_keys is None:
        requested = [
            {
                "model": model,
                "variant": variant or None,
                "representation": representation,
                "embedding_key": format_embedding_key(
                    embedding_template, variant, representation
                ),
            }
            for variant in variants
            for representation in representations
        ]
    else:
        requested = [
            {
                "model": model,
                "variant": None,
                "representation": "custom",
                "embedding_key": key,
            }
            for key in embedding_keys
        ]

    for item in requested:
        emb_key = item["embedding_key"]
        if emb_key not in adata.obsm:
            msg = f"{dataset}: missing obsm['{emb_key}']; skipping"
            if skip_missing:
                print(msg, flush=True)
                continue
            raise KeyError(msg)

        basis = umap_key_for(emb_key)
        if basis in adata.obsm and not overwrite:
            print(f"{dataset}: reusing obsm['{basis}']", flush=True)
        else:
            print(f"{dataset}: computing UMAP for obsm['{emb_key}']", flush=True)
            coords = compute_umap(
                adata,
                emb_key,
                random_state=random_state,
                min_dist=min_dist,
                n_neighbors=n_neighbors,
            )
            adata.obsm[basis] = coords
            write_obsm_array(h5ad_path, basis, coords)

        safe_key = emb_key.removeprefix("X_").replace("/", "_").replace(" ", "_")
        dataset_dir = output_dir_for(
            results_dir,
            output_dir_template,
            dataset=dataset,
            model=model,
            output_prefix=output_prefix,
            item=item,
        )
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.add(dataset_dir)
        for color in plot_colors:
            fig_path = dataset_dir / f"{dataset}_{safe_key}_umap_by_{color}.png"
            save_embedding_plot(
                adata,
                basis=basis,
                color=color,
                output_path=fig_path,
                size=size,
                alpha=alpha,
                dpi=dpi,
            )
            outputs.append(
                {
                    **item,
                    "umap_key": basis,
                    "color": color,
                    "figure": str(fig_path),
                }
            )

    manifest = {
        "dataset": dataset,
        "model": model,
        "h5ad_path": str(h5ad_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_template": embedding_template,
        "embedding_keys": embedding_keys,
        "output_prefix": output_prefix,
        "output_dir_template": output_dir_template,
        "random_state": random_state,
        "min_dist": min_dist,
        "n_neighbors": n_neighbors,
        "size": size,
        "alpha": alpha,
        "dpi": dpi,
        "outputs": outputs,
    }
    if not output_dirs:
        output_dirs.add(results_dir / output_prefix / dataset)
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{dataset}_{output_prefix}_umaps_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAPs for foundation-model embeddings.")
    parser.add_argument("--dataset", choices=[*sorted(DATASETS), "all"], default="dicty")
    parser.add_argument(
        "--model",
        choices=[*sorted(MODEL_PRESETS), "all"],
        default="main",
        help="Embedding preset to plot. Default 'main' = the repo-wide MAIN set.",
    )
    parser.add_argument("--variants", default=None)
    parser.add_argument(
        "--representations",
        default=None,
        help="Comma-separated: embedding,pca50",
    )
    parser.add_argument(
        "--embedding-template",
        default=None,
        help=(
            "obsm key template. Available fields: {variant}, {representation}, "
            "{representation_suffix}. Default: X_geneformer_ortholog_{variant}{representation_suffix}"
        ),
    )
    parser.add_argument(
        "--embedding-keys",
        default=None,
        help="Comma-separated explicit obsm keys. If set, variants/template are ignored.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Subdirectory/manifest prefix under results-dir.",
    )
    parser.add_argument(
        "--output-dir-template",
        default=None,
        help=(
            "Directory template under results-dir. Fields: dataset, model, "
            "output_prefix, variant, representation, embedding_key, layer."
        ),
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="Comma-separated obs columns. Defaults to dataset-specific time,cell_type.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output root. Defaults to the model preset, or results/embeddings/umaps.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (umap-learn default; matches embed_UCE notebook).",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--size", type=float, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", dest="skip_missing", action="store_true", default=True)
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    colors = parse_csv(args.colors) if args.colors else None

    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    models = sorted(MODEL_PRESETS) if args.model == "all" else [args.model]
    for model in models:
        preset = MODEL_PRESETS[model]
        variants = parse_csv(args.variants) if args.variants else list(preset.get("variants", ("",)))
        representations = (
            parse_csv(args.representations)
            if args.representations
            else list(preset.get("representations", ("embedding",)))
        )
        unknown_repr = [r for r in representations if r not in REPRESENTATION_SUFFIXES]
        if unknown_repr:
            raise ValueError("Unknown representations; choose from embedding,pca50")

        embedding_template = (
            args.embedding_template
            or preset.get("template")
            or DEFAULT_EMBEDDING_TEMPLATE
        )
        if args.embedding_keys:
            embedding_keys = parse_csv(args.embedding_keys)
        elif "keys" in preset and args.embedding_template is None:
            embedding_keys = list(preset["keys"])
        else:
            embedding_keys = None
        output_prefix = args.output_prefix or preset.get("output_prefix") or model
        output_dir_template = (
            args.output_dir_template
            or preset.get("output_dir_template")
            or "{output_prefix}/{dataset}"
        )
        results_dir = Path(args.results_dir or preset.get("results_dir") or DEFAULT_RESULTS_DIR)

        for dataset in datasets:
            run_dataset(
                dataset=dataset,
                model=model,
                variants=variants,
                representations=representations,
                embedding_template=embedding_template,
                embedding_keys=embedding_keys,
                output_prefix=output_prefix,
                output_dir_template=output_dir_template,
                colors=colors,
                results_dir=results_dir,
                random_state=args.random_state,
                min_dist=args.min_dist,
                n_neighbors=args.n_neighbors,
                size=args.size,
                alpha=args.alpha,
                dpi=args.dpi,
                overwrite=args.overwrite,
                skip_missing=args.skip_missing,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
