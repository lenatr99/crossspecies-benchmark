"""Apply one reproducible PCA reduction to frozen headline embeddings.

This publication sensitivity deliberately starts from the stored cell
representations rather than rerunning any embedding model.  Each source array
is centered (not feature-scaled) and reduced independently to the requested
number of principal components with the same sklearn implementation and seed.
The output keys and a hash-bearing manifest are additive: headline keys and
selected result runs are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata
import h5py
import numpy as np
import sklearn
from anndata._io.specs import write_elem
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from embedding_sets import DIMENSION_MATCHED_100_SUPP, DIMENSION_REDUCTION_SOURCES_100


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "dicty": ROOT / "data" / "dicty.h5ad",
    "elegans": ROOT / "data" / "elegans.h5ad",
}
DEFAULT_MANIFEST_DIR = ROOT / "results" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-cells",
        type=int,
        default=0,
        help="Fit twice on the first N cells, verify identical hashes, and do not write H5AD outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def load_obsm(path: Path, key: str, n_cells: int = 0) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "obsm" not in handle or key not in handle["obsm"]:
            raise KeyError(f"{path} is missing obsm[{key!r}]")
        dataset = handle["obsm"][key]
        if dataset.ndim != 2:
            raise ValueError(f"obsm[{key!r}] is not a matrix: {dataset.shape}")
        stop = min(n_cells, dataset.shape[0]) if n_cells > 0 else dataset.shape[0]
        return np.asarray(dataset[:stop], dtype=np.float32)


def write_obsm(path: Path, key: str, value: np.ndarray, overwrite: bool) -> None:
    with h5py.File(path, "a") as handle:
        obsm = handle.require_group("obsm")
        if key in obsm:
            if not overwrite:
                raise FileExistsError(f"{path}: obsm[{key!r}] already exists; pass --overwrite")
            del obsm[key]
        write_elem(obsm, key, np.asarray(value, dtype=np.float32))


def reduce(source: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, PCA]:
    if source.shape[0] <= n_components or source.shape[1] < n_components:
        raise ValueError(
            f"Cannot compute {n_components} PCs from array with shape {source.shape}"
        )
    estimator = PCA(
        n_components=n_components,
        svd_solver="randomized",
        whiten=False,
        random_state=seed,
    )
    output = estimator.fit_transform(source).astype(np.float32, copy=False)
    if output.shape != (source.shape[0], n_components):
        raise AssertionError(output.shape)
    if not np.isfinite(output).all():
        raise ValueError("PCA output contains non-finite values")
    if not np.all(np.var(output, axis=0) > 0):
        raise ValueError("PCA output contains a zero-variance component")
    return output, estimator


def run_dataset(dataset: str, args: argparse.Namespace) -> Path:
    path = DATASETS[dataset]
    records = []
    smoke = args.smoke_cells > 0
    for source_key, output_key in zip(
        DIMENSION_REDUCTION_SOURCES_100, DIMENSION_MATCHED_100_SUPP, strict=True
    ):
        source = load_obsm(path, source_key, args.smoke_cells)
        output, estimator = reduce(source, args.n_components, args.seed)
        output_hash = sha256_array(output)
        repeat_hash = None
        if smoke:
            repeated, _ = reduce(source, args.n_components, args.seed)
            repeat_hash = sha256_array(repeated)
            if repeat_hash != output_hash or not np.array_equal(output, repeated):
                raise AssertionError(f"Non-deterministic PCA output for {dataset}/{source_key}")
        else:
            write_obsm(path, output_key, output, args.overwrite)

        records.append(
            {
                "source_key": source_key,
                "output_key": output_key,
                "source_shape": list(source.shape),
                "source_sha256": sha256_array(source),
                "output_shape": list(output.shape),
                "output_sha256": output_hash,
                "repeat_output_sha256": repeat_hash,
                "explained_variance_ratio_sum": float(estimator.explained_variance_ratio_.sum()),
                "finite": True,
                "all_components_nonzero_variance": True,
            }
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "h5ad": str(path),
        "mode": "smoke" if smoke else "full",
        "smoke_cells": int(args.smoke_cells),
        "parameters": {
            "operation": "PCA applied after cell embedding",
            "n_components": int(args.n_components),
            "center": True,
            "feature_scaling": False,
            "whiten": False,
            "svd_solver": "randomized",
            "seed": int(args.seed),
        },
        "software_versions": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "anndata": anndata.__version__,
        },
        "records": records,
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    manifest_path = args.manifest_dir / f"dimension_matched_100_{dataset}{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return manifest_path


def main() -> int:
    args = parse_args()
    if args.n_components != 100:
        raise ValueError("This publication preset is fixed to 100 components")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
