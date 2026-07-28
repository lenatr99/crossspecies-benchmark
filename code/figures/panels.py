"""Low-level matplotlib primitives for the manuscript pseudotime figures.

Two families live here:

* UMAP scatter panels (``draw_continuous`` / ``draw_categorical``), which draw
  onto a caller-supplied ``Axes`` so the figure setup stays in one place; and
* self-contained chart builders (``plot_line``, ``plot_bioconservation``,
  ``plot_pseudotime_metrics``, ``plot_legend``) that create, save, and close
  their own figure.

The shared look is a fat black "halo" drawn under a coloured point cloud: a
first scatter with a thick black edge, then the same points again on top
without an edge. Point clouds are rasterised so the exported SVGs stay small
while axes/text remain vector.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure_presets import (
    BAR_PALETTE,
    CONTINUOUS_CMAP,
    DPI,
    FIGSIZE,
    GREY,
    HALO_CMAP,
    MARKER_SIZE,
    OUTLINE_WIDTH,
)

plt.rcParams["svg.fonttype"] = "none"  # keep text editable in the SVG
plt.rcParams["font.family"] = "Arial"
# With svg.fonttype="none" the family name ("Arial") is written into the SVG as
# text, so it renders correctly wherever Arial exists. If Arial isn't installed
# for matplotlib's own layout pass it silently falls back — quiet the per-glyph
# "findfont: Font family 'Arial' not found" spam.
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ---- UMAP scatter panels ------------------------------------------------
def new_umap_axes():
    """A blank square figure with the axes turned off, ready for a point cloud."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    return fig, ax


def _halo(ax, coords, *, c, cmap, size):
    """Thick black-edged scatter that sits under the coloured points."""
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=size,
        c=c,
        cmap=cmap,
        alpha=1,
        edgecolors="black",
        linewidth=OUTLINE_WIDTH,
    )


def draw_continuous(ax, coords, values, *, cmap=CONTINUOUS_CMAP, size=MARKER_SIZE, shuffle=False, seed=0):
    """Colour every cell by a continuous value (pseudotime or numeric time)."""
    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)
    order = np.arange(coords.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    xy, c = coords[order], values[order]
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=c, cmap=cmap, alpha=1,
               edgecolors="black", linewidth=OUTLINE_WIDTH, zorder=1, rasterized=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=c, cmap=cmap, alpha=1, zorder=2, rasterized=True)


def draw_categorical(ax, coords, codes, code_colors, *, size=MARKER_SIZE, halo_size=None,
                     grey=GREY, shuffle=False, seed=0):
    """Colour labelled cells by category; unlabelled cells (code ``-1``) go grey.

    ``codes`` are integer category codes (pandas ``.cat.codes``); ``code_colors``
    is a list of hex colours indexed by code. The order of the foreground points
    can be shuffled so no single category is always drawn on top.
    """
    coords = np.asarray(coords)
    codes = np.asarray(codes)
    _halo(ax, coords, c=codes, cmap=HALO_CMAP, size=halo_size or size)

    is_nan = codes < 0
    ax.scatter(coords[is_nan, 0], coords[is_nan, 1], s=size, c=grey,
               alpha=1, zorder=2, rasterized=True)

    fg = np.where(~is_nan)[0]
    if shuffle:
        np.random.default_rng(seed).shuffle(fg)
    fg_colors = [code_colors[code] for code in codes[fg]]
    ax.scatter(coords[fg, 0], coords[fg, 1], s=size, c=fg_colors,
               alpha=1, zorder=3, rasterized=True)


def save_panel(fig, path: Path, *, tight: bool = True):
    """Rasterise the point clouds, save the SVG, and close the figure."""
    for coll in fig.axes[0].collections:
        coll.set_rasterized(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight" if tight else None)
    plt.close(fig)


def plot_legend(categories, code_colors, path: Path):
    """Standalone legend for a categorical panel (one swatch per category)."""
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=code_colors[i],
               markersize=8, label=cat)
        for i, cat in enumerate(categories)
    ]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.legend(handles=handles, loc="center", frameon=False)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---- 1D pseudotime line plot --------------------------------------------
def plot_line(means, n_colors, path: Path, *, xticks, xlim, tick_labels=None):
    """Markers at the mean ground-truth time of each pseudotime bin."""
    colors = [cm.get_cmap("Spectral", n_colors)(i) for i in range(n_colors)]
    fig, ax = plt.subplots(figsize=(3, 0.4), dpi=DPI)
    for i, x in enumerate(means):
        ax.plot(x, 0, zorder=100, clip_on=False, marker="o", linestyle="None",
                markersize=15, markerfacecolor=mcolors.to_rgba(colors[i % len(colors)], alpha=1),
                markeredgewidth=1.5, markeredgecolor=(0, 0, 0, 1))

    ax.set(yticks=[], xticks=xticks, xlim=xlim, ylim=(0, 0.7))
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", labelsize=20, length=15, width=2)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_clip_on(False)
    ax.spines["bottom"].set_linewidth(2)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.60)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---- Horizontal grouped bar charts --------------------------------------
def _grouped_barh(scores, *, row_labels, methods, method_labels, palette,
                  xerr=None, value_fmt="{:.2f}", figsize=(6, 5)):
    """Shared layout for the avgBIO and pseudotime bar charts.

    ``scores`` is ``(n_rows, n_methods)``; rows go on the y axis, methods are the
    grouped bars within each row. ``xerr`` is an optional
    ``(2, n_rows, n_methods)`` array of asymmetric error bars.
    """
    n_rows, n_methods = scores.shape
    y = np.arange(n_rows)
    bar_h = 0.6 / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_h

    fig, ax = plt.subplots(figsize=figsize)
    for j in range(n_methods):
        ax.barh(
            y + offsets[j], scores[:, j], height=bar_h,
            xerr=None if xerr is None else [xerr[0, :, j], xerr[1, :, j]],
            color=palette[j % len(palette)], edgecolor="black",
            capsize=1.5, error_kw={"linewidth": 0.8},
        )
        for i in range(n_rows):
            x_val = scores[i, j]
            y_pos = y[i] + offsets[j]
            white_text = j < 3
            ax.text(x_val / 2, y_pos + 0.01, value_fmt.format(x_val),
                    va="center", ha="center", fontsize=8,
                    color="white" if white_text else "black")
            ax.text(x_val + 0.05, y_pos, method_labels[j], va="center", ha="left",
                    fontsize=8, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, rotation=90, va="center")
    ax.set_xlabel("Normalized score")
    ax.margins(y=0.06)
    ax.invert_yaxis()
    ax.set_xlim(left=-0.03, right=1.03)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_bioconservation(data, gene_sets, gene_set_labels, methods, method_labels, path: Path):
    """Grouped bar chart of avgBIO scores (gene sets on y, methods grouped)."""
    scores = np.array([
        [_avg_bio(data[gs][m]) for m in methods]
        for gs in gene_sets
    ])
    fig = _grouped_barh(
        scores, row_labels=gene_set_labels, methods=methods,
        method_labels=method_labels, palette=BAR_PALETTE, figsize=(6, 5),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pseudotime_metrics(point_data, stability_data, metric_keys, metric_labels, methods, method_labels, path: Path):
    """Plot full-data scores with repeated-subsampling stability intervals."""
    def norm(r):
        return (r + 1) / 2

    scores = np.array([[norm(point_data[m][k]) for m in methods] for k in metric_keys])
    lo = np.array([[norm(stability_data[m][f"{k}_ci"][0]) for m in methods] for k in metric_keys])
    hi = np.array([[norm(stability_data[m][f"{k}_ci"][1]) for m in methods] for k in metric_keys])
    # Clip at 0: a percentile CI can occasionally land on the wrong side of the
    # point estimate, and matplotlib rejects negative error-bar lengths.
    xerr = np.clip(np.stack([scores - lo, hi - scores]), 0, None)  # (2, n_metrics, n_methods)

    fig = _grouped_barh(
        scores, row_labels=metric_labels, methods=methods,
        method_labels=method_labels, palette=BAR_PALETTE, xerr=xerr, figsize=(6, 4),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _avg_bio(method_entry):
    """avgBIO may be a bare float or a ``{"mean": ...}`` dict; return the float."""
    val = method_entry["avg_bio"]
    return float(val["mean"]) if isinstance(val, dict) else float(val)


# ---- Genome-coverage panel ----------------------------------------------
def plot_genome_coverage(rows, total, path: Path):
    """Horizontal-bar summary of how many dataset genes each method uses.

    ``rows`` is an ordered list of ``(label, covered, required_mapping,
    retained_subset)`` (top row drawn first). Bars show ``covered / total`` as a
    percentage; the annotation columns to the right of the axis carry the raw
    gene count and the descriptive mapping/subset text.
    """
    labels = [r[0] for r in rows]
    covered = [int(r[1]) for r in rows]
    required = [r[2] for r in rows]
    retained = [r[3] for r in rows]
    pct = [100.0 * c / total if total else 0.0 for c in covered]

    # Annotation columns live in data coordinates to the right of the bar axis
    # (which spans 0-100). Widening the figure stretches the bar axis, so the
    # extra width lands on the bars; the columns are anchored at the same
    # physical offset past the bar end (x=100) so the right block doesn't drift.
    X_COUNT = 126.7   # raw gene-count column (centred)
    X_TEXT = 144.0    # mapping/subset column (left-aligned); pulled in ~0.5" to trim width

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 6))  # width -0.5" to narrow the bar plot

    bar_color = "#555555"
    ax.barh(y, pct, height=0.55, color=bar_color, edgecolor="black",
            linewidth=2, zorder=3, clip_on=False)
    for yi, v in zip(y, pct):
        ax.text(v / 2, yi, f"{v:.1f}%", va="center", ha="center",
                color="#ffffff", fontsize=14, zorder=4)

    # Raw gene count column.
    for yi, c in zip(y, covered):
        ax.text(X_COUNT, yi, f"{c:,}", va="center", ha="center", color="black",
                fontsize=14, clip_on=False)

    # Required-mapping above retained-subset (bold) in a single column; SUB_DY is
    # the gap between the two stacked lines (also used by the headers).
    SUB_DY = 0.28
    for yi, req in zip(y, required):
        ax.text(X_TEXT, yi - SUB_DY, req, va="center", ha="left", color="black",
                fontsize=14, clip_on=False)
    for yi, ret in zip(y, retained):
        ax.text(X_TEXT, yi, ret, va="center", ha="left", color="black",
                fontweight="bold", fontsize=14, clip_on=False)

    # Column headers (drawn above the first row).
    ax.text(50, -1, "Coverage", clip_on=False, va="center", ha="center", fontsize=14)
    ax.text(X_COUNT, -1, "Genes (n)", clip_on=False, va="center", ha="center", fontsize=14)
    ax.text(X_TEXT, -1 - SUB_DY, "Required mapping", clip_on=False, va="center", ha="left", fontsize=14)
    ax.text(X_TEXT, -1, "Retained gene subset", clip_on=False, va="center", ha="left",
            fontweight="bold", fontsize=14)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"], fontsize=16)
    ax.set_xlabel("% of dataset genes", labelpad=20, fontsize=14)
    ax.set_xlim(left=-5, right=103)  # extra ~0.1" pad to the left of the bars
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis="both", width=1.5, length=8)
    ax.invert_yaxis()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
