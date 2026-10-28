"""Redraw the Supplementary Fig. S2 D. discoideum UCE UMAPs in the main-figure style.

The panels shipped in the manuscript were previously raw ``sc.pl.embedding`` output
(scanpy default colours, a "cell_type" title and an inline legend), which looked
nothing like the UMAP panels in the main figures. This script reuses the exact
main-figure primitive (``panels.draw_categorical``: a black halo under a coloured
point cloud, prespore/prestalk in the manuscript palette, grey unlabelled cells)
so S2 matches Fig. 1g.  In particular, every panel uses the same fixed canvas
and display aspect as the main-figure UMAPs.

Reads the pre-computed UMAP coordinates already stored in ``data/dicty.h5ad`` (no
re-embedding); writes clean panel PNGs plus a publication-ready composite with
Graphik Trial titles and legend to ``manuscript/figures/supplementary/``.

Run (figures env):
  /opt/anaconda3/envs/crossspecies-foundation-analysis/bin/python code/figures/make_s2_umaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "figures"))

import anndata as ad  # noqa: E402

import panels  # noqa: E402
from figure_presets import DATASETS, DPI  # noqa: E402
from make_figures import category_colors, reordered_codes  # noqa: E402

OUT = ROOT / "manuscript" / "figures" / "supplementary"
FONT_DIR = ROOT / "fonts"
GRAPHIK_REGULAR = FONT_DIR / "Graphik-Regular-Trial.otf"
GRAPHIK_SEMIBOLD = FONT_DIR / "Graphik-Semibold-Trial.otf"

# (UMAP basis, output filename stem, rotate 180 degrees).  The UCE-4L panel
# deliberately uses the exact basis rendered in Figure 1, rotated 180 degrees
# for the supplemental comparison; the stored sensitivity UMAPs retain their
# established display orientation.
VARIANTS = [
    ("X_uce_4L_umap", "s2_uce_4L_uncalibrated", True),
    ("X_uce_33L_50_umap", "s2_uce_33L_uncalibrated", True),
    ("X_uce_33L_whitening_50_umap", "s2_uce_33L_whitening", True),
    ("X_uce_33L_macrogene_50_umap", "s2_uce_33L_macrogene", True),
]
TITLES = (
    "UCE-4L",
    "UCE-33L",
    "UCE-33L, whitening",
    "UCE-33L, macrogene",
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = DATASETS["dicty"]["categorical_panels"]["cell_type"]

    adata = ad.read_h5ad(DATASETS["dicty"]["data"], backed="r")
    present = adata.obs[panel["obs_key"]].astype("category").cat.categories
    categories, colors = category_colors(panel, present)
    codes = reordered_codes(adata.obs[panel["obs_key"]], categories)

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.lines import Line2D

    missing_fonts = [path for path in (GRAPHIK_REGULAR, GRAPHIK_SEMIBOLD) if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError("Missing Graphik Trial fonts: " + ", ".join(map(str, missing_fonts)))
    for font_path in (GRAPHIK_REGULAR, GRAPHIK_SEMIBOLD):
        fontManager.addfont(str(font_path))
    graphik = FontProperties(fname=GRAPHIK_REGULAR, size=10)
    graphik_semibold = FontProperties(fname=GRAPHIK_SEMIBOLD, size=10)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

    panel_paths = []
    for basis, stem, rotate_180 in VARIANTS:
        if basis not in adata.obsm:
            raise KeyError(f"missing obsm['{basis}'] in {DATASETS['dicty']['data']}")
        coords = np.asarray(adata.obsm[basis])
        if rotate_180:
            coords = -coords
        # Exactly the main-figure recipe (panels.save_panel): blank fixed canvas,
        # draw the halo + point cloud, and let Matplotlib use the same display
        # aspect as every UMAP in Figs. 1 and 2.  Do not impose ``aspect='equal'``
        # here: that produced four supplemental bitmaps with markedly different
        # heights and made the one-row comparison look compressed.
        fig, ax = panels.new_umap_axes()
        panels.draw_categorical(
            ax, coords, codes, colors,
            halo_size=panel.get("halo_size"), shuffle=panel.get("shuffle", False),
        )
        for coll in ax.collections:
            coll.set_rasterized(True)
        out_path = OUT / f"{stem}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=True)
        plt.close(fig)
        panel_paths.append(out_path)
        print(f"  wrote {out_path.relative_to(ROOT)}", flush=True)

    # Compose the four rasterized point clouds with vector Graphik text.  Using
    # the already rendered panel images keeps their point/halo proportions
    # identical to the main figures while allowing a wider inter-panel gutter.
    composite, axes = plt.subplots(1, 4, figsize=(8.8, 2.7))
    for ax, title, path in zip(axes, TITLES, panel_paths):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontproperties=graphik_semibold, pad=7)
        ax.set_axis_off()

    legend_colors = [*colors, "#8A8A8A"]
    legend_labels = [*categories, "unlabelled"]
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=color, markeredgecolor="none", label=label,
        )
        for label, color in zip(legend_labels, legend_colors)
    ]
    composite.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        columnspacing=2.3,
        handletextpad=0.55,
        frameon=False,
        prop=graphik,
    )
    composite.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.23, wspace=0.18)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        out_path = OUT / f"s2_uce_checkpoint_calibration.{suffix}"
        composite.savefig(out_path, bbox_inches="tight", transparent=True, **kwargs)
        print(f"  wrote {out_path.relative_to(ROOT)}", flush=True)
    plt.close(composite)

    print("categories:", list(categories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
