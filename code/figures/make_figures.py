"""Render the manuscript pseudotime / trajectory figures from a dataset H5AD.

For each dataset this produces, per method:

* ``pseudotime/<method>.svg``  – UMAP coloured by DPT pseudotime
* ``time/<method>.svg``        – UMAP coloured by ground-truth time
* ``<gene_set>/<method>.svg``  – UMAP coloured by each categorical gene set
* ``line_plot/1dplot_pseudotime_<method>.svg`` – mean time per pseudotime bin

and once per dataset:

* ``<gene_set>/legend.svg``                – legend for panels that request one
* ``bioconservation/bioconservation_metrics.svg`` – grouped avgBIO bar chart
* ``pseudotime/pseudotime_metrics.svg``    – grouped pseudotime-metric bar chart
* ``genome_coverage/genome_coverage_<organism>.svg`` – per-method genome coverage
  (gene counts read from the embedding manifests)

The H5AD is opened in backed mode, so the (multi-GB) expression matrix stays on
disk; only ``obs`` and the UMAP ``obsm`` arrays are read into memory.

Examples
--------
    python code/figures/make_figures.py --dataset dicty
    python code/figures/make_figures.py --dataset dicty --only pseudotime,time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE_DIR = ROOT / "data" / "evaluation" / ".cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(DEFAULT_CACHE_DIR / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import panels
from figure_presets import (
    DATASETS,
    GENOME_COVERAGE_META,
    METHOD_LABELS,
    METHODS,
    PSEUDOTIME_RUNS,
    avgbio_dir,
    embeddings_dir,
    output_root,
    pseudotime_dir,
    selected_results_file,
)

PSEUDOTIME_METRICS = {
    "spearman": "Spearman",
    "lins_ccc_consensus": "Consensus Lin's CCC",
}


# ---- Helpers ------------------------------------------------------------
def parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def numeric_time(series: pd.Series) -> np.ndarray:
    """Pull the first number out of each time label (``"04hr"`` -> ``4.0``)."""
    if series.dtype.kind in "iuf":
        return series.to_numpy(dtype=float)
    return (
        series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float).to_numpy()
    )


def category_colors(panel: dict, present: pd.Index) -> tuple[list[str], list[str]]:
    """Resolve a categorical panel spec to (ordered categories, colors-by-code).

    Categories absent from the data are dropped; the returned colour list is
    indexed by the (possibly reordered) category code. ``panel["colors"]`` may be
    a ``{category: hex}`` dict (named) or a list of hex colours positional in
    category order.
    """
    order = panel.get("order") or list(present)
    categories = [c for c in order if c in set(present)]
    colors_spec = panel["colors"]
    if isinstance(colors_spec, dict):
        colors = [colors_spec[c] for c in categories]
    else:  # positional list, one colour per category in order
        colors = list(colors_spec)[: len(categories)]
    return categories, colors


def reordered_codes(series: pd.Series, categories: list[str]) -> np.ndarray:
    """Codes for ``series`` under an explicit category order (missing -> -1)."""
    cat = pd.Categorical(series.astype(str), categories=categories)
    return cat.codes


def load_adata(path: Path):
    import anndata as ad

    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Reading {path} (backed)...", flush=True)
    return ad.read_h5ad(path, backed="r")


# ---- Per-family rendering -----------------------------------------------
def render_continuous_panels(adata, methods, *, value_for, out_dir, shuffle):
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        coords = adata.obsm[f"{method}_umap"]
        fig, ax = panels.new_umap_axes()
        panels.draw_continuous(ax, coords, value_for(method), shuffle=shuffle)
        panels.save_panel(fig, out_dir / f"{method}.svg")
        print(f"  wrote {out_dir.name}/{method}.svg", flush=True)


def render_categorical_panels(adata, methods, name, panel, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    present = adata.obs[panel["obs_key"]].astype("category").cat.categories
    categories, colors = category_colors(panel, present)
    codes = reordered_codes(adata.obs[panel["obs_key"]], categories)
    for method in methods:
        coords = adata.obsm[f"{method}_umap"]
        fig, ax = panels.new_umap_axes()
        panels.draw_categorical(
            ax, coords, codes, colors,
            halo_size=panel.get("halo_size"), shuffle=panel.get("shuffle", False),
        )
        panels.save_panel(fig, out_dir / f"{method}.svg")
        print(f"  wrote {name}/{method}.svg", flush=True)
    if panel.get("legend"):
        panels.plot_legend(categories, colors, out_dir / "legend.svg")
        print(f"  wrote {name}/legend.svg", flush=True)


def render_line_plots(adata, methods, line_values, out_dir, line_cfg, *, n_bins=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    xticks = line_cfg["xticks"]
    xlim = tuple(line_cfg["xlim"])
    tick_labels = line_cfg.get("tick_labels")
    time_series = pd.Series(line_values, index=adata.obs_names)
    for method in methods:
        pt = pd.Series(np.asarray(adata.obs[f"{method}_dpt_pseudotime"], float),
                       index=adata.obs_names)
        bins = pd.cut(pt, bins=n_bins)
        means = time_series.groupby(bins, observed=True).mean().tolist()
        path = out_dir / f"1dplot_pseudotime_{method.lower()}.svg"
        panels.plot_line(means, n_bins, path, xticks=xticks, xlim=xlim, tick_labels=tick_labels)
        print(f"  wrote line_plot/{path.name}", flush=True)


def render_bioconservation(organism, methods, gene_sets, out_dir):
    data = {}
    missing = []
    for gs in gene_sets:
        json_path = selected_results_file(avgbio_dir(organism), f"{gs}.json")
        if json_path is None:
            missing.append(f"{gs}.json")
            continue
        data[gs] = json.loads(json_path.read_text())
    if missing:
        print(f"  skipping bioconservation: no results for {', '.join(missing)} "
              f"(pin a run: xspecies select --dataset {organism} --kind avgbio --run latest)", flush=True)
        return
    # The grouped bar needs every method present in every gene-set JSON. If a
    # JSON was computed for a different embedding set, say which keys are absent
    # (re-run avgBIO for the MAIN set) rather than crashing with a KeyError.
    absent = {gs: [m for m in methods if m not in data[gs]] for gs in gene_sets}
    absent = {gs: ms for gs, ms in absent.items() if ms}
    if absent:
        for gs, ms in absent.items():
            print(f"  skipping bioconservation: {gs}.json has no {', '.join(ms)} "
                  f"(re-run: xspecies avgbio --dataset {organism})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    panels.plot_bioconservation(
        data, list(gene_sets), [gene_sets[gs] for gs in gene_sets],
        methods, [METHOD_LABELS.get(m, m) for m in methods],
        out_dir / "bioconservation_metrics.svg",
    )
    print("  wrote bioconservation/bioconservation_metrics.svg", flush=True)


def render_pseudotime_metrics(organism, methods, out_dir):
    run_ids = PSEUDOTIME_RUNS[organism]
    base = pseudotime_dir(organism) / "runs"
    point_path = base / run_ids["point"] / "pseudotime.json"
    stability_path = base / run_ids["stability"] / "pseudotime.json"
    for role, path in (("point", point_path), ("stability", stability_path)):
        if not path.is_file():
            print(
                f"  skipping pseudotime_metrics: missing manuscript {role} run {path}",
                flush=True,
            )
            return
    point_data = json.loads(point_path.read_text())
    stability_data = json.loads(stability_path.read_text())
    absent = [m for m in methods if m not in point_data or m not in stability_data]
    if absent:
        print(f"  skipping pseudotime_metrics: pseudotime.json has no {', '.join(absent)} "
              f"(re-run: xspecies pseudotime --dataset {organism})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    panels.plot_pseudotime_metrics(
        point_data, stability_data,
        list(PSEUDOTIME_METRICS), list(PSEUDOTIME_METRICS.values()),
        methods, [METHOD_LABELS.get(m, m) for m in methods],
        out_dir / "pseudotime_metrics.svg",
    )
    print("  wrote pseudotime/pseudotime_metrics.svg", flush=True)


# ---- Genome coverage (reads gene counts from the embedding manifests) ----
def _load_manifest(organism: str, model: str) -> dict | None:
    path = embeddings_dir() / f"{model}_{organism}_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def coverage_for_key(organism: str, key: str) -> tuple[int, int] | None:
    """(genes used, total dataset genes) for a MAIN embedding key.

    Both numbers come from the per-method embedding manifest, so the panel never
    drifts from what the embeddings actually used:

    * PCA / scVI  -> all dataset genes (``n_source_genes``);
    * scGPT / Geneformer ortholog-tier -> genes kept in that tier that also map
      into the model vocabulary (``n_unique_ids`` / ``n_geneformer_vocab_matched_genes``);
    * UCE -> protein-coding genes surviving the UCE asset filter (``n_validated_genes``).
    """
    kl = key.lower()

    if key.startswith("X_uce"):
        man = _load_manifest(organism, "uce")
        if not man:
            return None
        runs = man["runs"]  # coverage is depth-independent (protein-embedding filter)
        return max(r["n_validated_genes"] for r in runs), max(r["n_source_genes"] for r in runs)

    model = next((m for m in ("pca", "scvi", "scgpt", "geneformer") if m in kl), None)
    if model is None:
        return None
    man = _load_manifest(organism, model)
    if not man:
        return None
    variants = {v["variant"]: v for v in man["variants"]}
    total = next(iter(variants.values()))["n_source_genes"]

    variant = next((t for t in ("tier123", "tier12", "tier1") if t in kl), "full")
    if model in ("pca", "scvi") and variant == "full":
        full = variants.get("full")
        return (int(full["n_unique_ids"]) if full else total), total
    v = variants.get(variant)
    if v is None:
        return None
    # "genes available to the method" = ortholog-mapped genes that also land in the
    # model's token vocabulary. Geneformer records the vocab-matched count directly;
    # scGPT/others fall back to the unique ortholog-id count. ``or`` (not dict.get's
    # default) so a present-but-null field still falls through.
    covered = v.get("n_geneformer_vocab_matched_genes") or v.get("n_unique_ids")
    if covered is None:
        return None
    return int(covered), total


def render_genome_coverage(organism, methods, out_dir):
    rows = []
    total = None
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        meta = GENOME_COVERAGE_META.get(label)
        cov = coverage_for_key(organism, method)
        if meta is None or cov is None:
            if meta is not None:
                print(f"  skipping genome_coverage row {label}: no manifest counts "
                      f"(run: xspecies embed <model> --dataset {organism})", flush=True)
            continue
        covered, total = cov
        rows.append((label, covered, meta[0], meta[1]))
    if not rows:
        print("  skipping genome_coverage: no manifest gene counts found", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"genome_coverage_{organism}.svg"
    panels.plot_genome_coverage(rows, total, path)
    print(f"  wrote genome_coverage/{path.name}", flush=True)


# ---- Orchestration ------------------------------------------------------
def available_families(cfg: dict) -> list[str]:
    return [
        "pseudotime", "time", *cfg["categorical_panels"], "line_plot",
        "bioconservation", "pseudotime_metrics", "genome_coverage",
    ]


def _methods_with_obs(methods, adata, suffix, family):
    """Methods that have ``<method><suffix>`` in obs; warn about (and drop) the rest."""
    have = [m for m in methods if f"{m}{suffix}" in adata.obs.columns]
    missing = [m for m in methods if f"{m}{suffix}" not in adata.obs.columns]
    if missing:
        print(f"  [{family}] skipping methods missing {suffix}: {', '.join(missing)}", flush=True)
    return have


def run_dataset(organism: str, methods: list[str], only: set[str], out_base: Path) -> None:
    cfg = DATASETS[organism]
    adata = load_adata(cfg["data"])

    skipped = [m for m in methods if f"{m}_umap" not in adata.obsm]
    if skipped:
        print(f"  skipping methods without a UMAP basis: {', '.join(skipped)}", flush=True)
    methods = [m for m in methods if f"{m}_umap" in adata.obsm]

    org_dir = out_base / organism
    t_num = numeric_time(adata.obs[cfg["time_key"]])           # time UMAP colour
    line_cfg = cfg["line_plot"]
    line_values = (                                            # line-plot binning
        numeric_time(adata.obs[line_cfg["time_col"]]) if line_cfg["time_col"] else t_num
    )

    families = available_families(cfg) if not only else [f for f in available_families(cfg) if f in only]
    for family in families:
        print(f"[{organism}] {family}", flush=True)
        if family == "pseudotime":
            render_continuous_panels(
                adata, _methods_with_obs(methods, adata, "_dpt_pseudotime", family),
                value_for=lambda m: np.asarray(adata.obs[f"{m}_dpt_pseudotime"], float),
                out_dir=org_dir / "pseudotime", shuffle=False,
            )
        elif family == "time":
            render_continuous_panels(
                adata, methods, value_for=lambda m: t_num,
                out_dir=org_dir / "time", shuffle=True,
            )
        elif family in cfg["categorical_panels"]:
            render_categorical_panels(
                adata, methods, family, cfg["categorical_panels"][family],
                org_dir / family,
            )
        elif family == "line_plot":
            render_line_plots(
                adata, _methods_with_obs(methods, adata, "_dpt_pseudotime", family),
                line_values, org_dir / "line_plot", line_cfg,
            )
        elif family == "bioconservation":
            render_bioconservation(
                organism, methods, cfg["bioconservation_gene_sets"],
                org_dir / "bioconservation",
            )
        elif family == "pseudotime_metrics":
            render_pseudotime_metrics(organism, methods, org_dir / "pseudotime")
        elif family == "genome_coverage":
            render_genome_coverage(organism, methods, org_dir / "genome_coverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=[*sorted(DATASETS), "all"],
        default="all",
        help="Dataset(s) to render (default: all, i.e. both dicty and elegans).",
    )
    parser.add_argument("--methods", default="", help="Comma-separated obsm method prefixes. Empty uses the preset five.")
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma-separated subset of panel families to render (default: all). "
            "Shared families: pseudotime, time, line_plot, bioconservation, "
            "pseudotime_metrics, genome_coverage. Dataset-specific categorical panels: dicty -> "
            "cell_type, cAMP_genes, milestones; elegans -> cell_type, "
            "developmental_markers, lineage_markers."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Override the manuscript/figures/generated root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = parse_csv(args.methods) or list(METHODS)
    only = set(parse_csv(args.only))
    out_base = output_root(args.output_root)
    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for organism in datasets:
        run_dataset(organism, methods, only, out_base)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
