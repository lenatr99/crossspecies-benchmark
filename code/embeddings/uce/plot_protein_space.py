"""Reproducible raw-versus-calibrated ESM-2 protein-space visualization.

The script balances human, D. discoideum and C. elegans protein dictionaries,
uses identical target-species genes before and after global whitening, and fits
one shared PCA/UMAP so both panels use the same coordinates. The visualization
is descriptive; quantitative cosine diagnostics remain the primary evidence
for what the calibration changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "manuscript" / "figures" / "supplementary"
DEFAULT_CACHE = ROOT / "data" / "foundation_models" / ".cache"
DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE / "xdg"))

FONT_DIR = ROOT / "fonts"
GRAPHIK_REGULAR = FONT_DIR / "Graphik-Regular-Trial.otf"
GRAPHIK_ITALIC = FONT_DIR / "Graphik-RegularItalic-Trial.otf"
GRAPHIK_SEMIBOLD = FONT_DIR / "Graphik-Semibold-Trial.otf"

RAW_DIR = ROOT / "models" / "UCE" / "model_files" / "protein_embeddings"
RAW_PATHS = {
    "human": RAW_DIR / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    "dicty": RAW_DIR / "Dictyostelium_discoideum.dicty_2.7.pep.all.gene_symbol_to_embedding_ESM2.pt",
    "elegans": RAW_DIR / "Caenorhabditis_elegans.WBcel235.pep.all.gene_symbol_to_embedding_ESM2.pt",
}
CALIBRATED_PATHS = {
    "dicty": ROOT / "results" / "embeddings" / "uce" / "esm2_adapter" / "dicty_whitening_global"
    / "dicty_whitening_global_gene_symbol_to_embedding_ESM2_adapted.pt",
    "elegans": ROOT / "results" / "embeddings" / "uce" / "esm2_adapter" / "elegans_whitening_global"
    / "elegans_whitening_global_gene_symbol_to_embedding_ESM2_adapted.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty protein dictionary in {path}")
    return payload


def rows_for(payload: dict, genes: list[str]):
    import numpy as np
    import torch

    rows = []
    for gene in genes:
        value = payload[gene]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        rows.append(np.asarray(value, dtype=np.float32))
    return np.stack(rows)


def choose_genes(keys, n: int, rng) -> list[str]:
    import numpy as np

    ordered = np.asarray(sorted(str(k) for k in keys), dtype=object)
    if len(ordered) < n:
        raise ValueError(f"Requested {n} proteins but only {len(ordered)} are available")
    return sorted(ordered[rng.choice(len(ordered), size=n, replace=False)].tolist())


def l2_normalize(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def sha256_array(array) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-proteins", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-stem", default="protein_space_raw_calibrated")
    return parser.parse_args()


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import umap
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA

    args = parse_args()
    if args.max_proteins < 50:
        raise ValueError("--max-proteins must be at least 50")
    required = [*RAW_PATHS.values(), *CALIBRATED_PATHS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing protein assets: " + ", ".join(missing))
    missing_fonts = [
        str(path) for path in (GRAPHIK_REGULAR, GRAPHIK_ITALIC, GRAPHIK_SEMIBOLD)
        if not path.is_file()
    ]
    if missing_fonts:
        raise FileNotFoundError("Missing Graphik Trial fonts: " + ", ".join(missing_fonts))
    for font_path in (GRAPHIK_REGULAR, GRAPHIK_ITALIC, GRAPHIK_SEMIBOLD):
        fontManager.addfont(str(font_path))
    graphik_italic = FontProperties(fname=GRAPHIK_ITALIC, size=10)
    graphik_title = FontProperties(fname=GRAPHIK_SEMIBOLD, size=11)
    graphik_panel = FontProperties(fname=GRAPHIK_SEMIBOLD, size=12)

    rng = np.random.default_rng(args.seed)
    sampled: dict[str, list[str]] = {}
    arrays: dict[str, np.ndarray] = {}

    human = load_payload(RAW_PATHS["human"])
    sampled["human"] = choose_genes(human.keys(), args.max_proteins, rng)
    arrays["human"] = rows_for(human, sampled["human"])
    del human

    for organism in ("dicty", "elegans"):
        raw = load_payload(RAW_PATHS[organism])
        calibrated = load_payload(CALIBRATED_PATHS[organism])
        shared = set(map(str, raw)).intersection(map(str, calibrated))
        sampled[organism] = choose_genes(shared, args.max_proteins, rng)
        arrays[f"{organism}_raw"] = rows_for(raw, sampled[organism])
        arrays[f"{organism}_calibrated"] = rows_for(calibrated, sampled[organism])
        del raw, calibrated

    order = ("human", "dicty_raw", "elegans_raw", "dicty_calibrated", "elegans_calibrated")
    lengths = {name: len(arrays[name]) for name in order}
    matrix = l2_normalize(np.vstack([arrays[name] for name in order]).astype(np.float32, copy=False))
    n_components = min(args.n_components, matrix.shape[0] - 1, matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    projected = pca.fit_transform(matrix).astype(np.float32, copy=False)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.seed,
        low_memory=True,
    )
    coords = reducer.fit_transform(projected).astype(np.float32, copy=False)

    spans: dict[str, slice] = {}
    start = 0
    for name in order:
        spans[name] = slice(start, start + lengths[name])
        start += lengths[name]

    # Match the organism colours used throughout the manuscript UMAP panels.
    # Human is a neutral reference rather than a third competing accent colour.
    colors = {"human": "#8A8A8A", "dicty": "#F46D43", "elegans": "#3882B9"}
    labels = {
        "human": "H. sapiens",
        "dicty": "D. discoideum",
        "elegans": "C. elegans",
    }
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.15))
    panel_specs = (
        ("a", "Raw ESM-2 embeddings", "dicty_raw", "elegans_raw"),
        ("b", "After global moment matching", "dicty_calibrated", "elegans_calibrated"),
    )
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    x_pad = max((x_max - x_min) * 0.02, 1e-6)
    y_pad = max((y_max - y_min) * 0.02, 1e-6)
    for ax, (panel_letter, title, dicty_key, elegans_key) in zip(axes, panel_specs):
        for label, key in (("human", "human"), ("dicty", dicty_key), ("elegans", elegans_key)):
            xy = coords[spans[key]]
            is_reference = label == "human"
            ax.scatter(
                xy[:, 0], xy[:, 1],
                s=2.0 if is_reference else 2.8,
                alpha=0.38 if is_reference else 0.68,
                linewidths=0,
                color=colors[label],
                rasterized=True,
                zorder=1 if is_reference else 2,
            )
        ax.set_title(title, fontproperties=graphik_title, pad=8)
        ax.text(
            0.0, 1.02, panel_letter, transform=ax.transAxes,
            fontproperties=graphik_panel, ha="left", va="bottom",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=colors[label], markeredgecolor="none",
            label=labels[label],
        )
        for label in ("human", "dicty", "elegans")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        columnspacing=2.3,
        handletextpad=0.55,
        frameon=False,
        prop=graphik_italic,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.89, bottom=0.20, wspace=0.08)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = args.output_dir / f"{args.output_stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", transparent=suffix != "png", **kwargs)
        outputs.append(path)
    plt.close(fig)

    sample_path = args.output_dir / f"{args.output_stem}_sampled_genes.tsv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["organism", "gene"])
        for organism in ("human", "dicty", "elegans"):
            writer.writerows((organism, gene) for gene in sampled[organism])

    coordinate_path = args.output_dir / f"{args.output_stem}_coordinates.npz"
    np.savez_compressed(
        coordinate_path,
        human=coords[spans["human"]],
        dicty_raw=coords[spans["dicty_raw"]],
        elegans_raw=coords[spans["elegans_raw"]],
        dicty_calibrated=coords[spans["dicty_calibrated"]],
        elegans_calibrated=coords[spans["elegans_calibrated"]],
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_proteins_per_species": args.max_proteins,
        "n_components": n_components,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "normalization": "row_l2_before_shared_pca",
        "projection": "single_shared_pca_and_umap_over_human_plus_raw_and_calibrated_targets",
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "input_sha256": {str(path): sha256_file(path) for path in required},
        "sampled_genes": str(sample_path),
        "same_target_identifiers_before_and_after": True,
        "human_coordinates_reused_between_panels": True,
        "shared_axis_limits": {
            "x": [x_min - x_pad, x_max + x_pad],
            "y": [y_min - y_pad, y_max + y_pad],
        },
        "human_coordinates_sha256": sha256_array(coords[spans["human"]]),
        "visual_style": {
            "regular_font": str(GRAPHIK_REGULAR),
            "italic_font": str(GRAPHIK_ITALIC),
            "semibold_font": str(GRAPHIK_SEMIBOLD),
            "colors": colors,
            "legend_frame": False,
            "legend_marker_size_points": 8,
        },
        "coordinate_file": str(coordinate_path),
        "outputs": [str(path) for path in outputs],
        "coordinate_shape": list(coords.shape),
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "torch": importlib.metadata.version("torch"),
            "umap-learn": importlib.metadata.version("umap-learn"),
        },
    }
    manifest_path = args.output_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
