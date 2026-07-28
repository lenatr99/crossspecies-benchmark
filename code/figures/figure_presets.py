"""Editable per-organism configuration for the manuscript pseudotime figures.

Everything that differs between datasets (data path, ground-truth time column,
the categorical "gene set" panels and their colours, the bar-chart inputs) is
declared here so :mod:`make_figures` stays generic. Add an organism by adding
an entry to :data:`DATASETS`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Pull the headline methods from the repo-wide single source of truth so the
# figures compare exactly the same embeddings as avgBIO / pseudotime.
sys.path.insert(0, str(ROOT / "code"))
from embedding_sets import MAIN as METHODS  # noqa: E402
from embedding_sets import MAIN_LABELS as METHOD_LABELS  # noqa: E402

# ---- Shared visual style ------------------------------------------------
MARKER_SIZE = 45
OUTLINE_WIDTH = 11          # width of the black halo drawn under each panel
CONTINUOUS_CMAP = "Spectral"
HALO_CMAP = "Spectral_r"    # only used to give categorical panels their outline
GREY = "#efefef"            # background colour for NaN / unlabelled cells
FIGSIZE = (7, 6.5)
DPI = 300

# The five "headline" methods (METHODS / METHOD_LABELS) come from
# code/embedding_sets.py — see the import at the top of this file.

# Greyscale ramp used for the grouped bar charts (one shade per method).
BAR_PALETTE = ("#555555", "#7b7b7b", "#a2a1a2", "#c8c7c8", "#eeedee")

# Manuscript figures combine full-data point estimates with intervals from the
# completed repeated-subsampling run. Keep these run IDs explicit so a CURRENT
# pointer cannot silently mix bootstrap intervals into a publication panel.
PSEUDOTIME_RUNS = {
    "dicty": {"point": "20260703_130811", "stability": "20260709_222558"},
    "elegans": {"point": "20260703_130811", "stability": "20260709_222558"},
}

# Per-method descriptive columns for the genome-coverage panel. Keyed by the
# display label (MAIN_LABELS), each entry is (required mapping, retained gene
# subset). These describe *how* each method restricts the gene set; the actual
# gene counts are read from the embedding manifests, never hard-coded.
GENOME_COVERAGE_META = {
    "PCA": ("gene ID", "all genes"),
    "scVI": ("gene ID", "all genes"),
    "scGPT": ("gene ID → human ortholog", "mapped orthologs"),
    "Geneformer": ("gene ID → human ortholog", "mapped orthologs"),
    "UCE-4L": ("gene ID → protein sequence", "protein-coding genes"),
}

_MISSING_COVERAGE_META = set(METHOD_LABELS.values()) - set(GENOME_COVERAGE_META)
if _MISSING_COVERAGE_META:
    raise RuntimeError(
        "Missing genome-coverage metadata for headline method label(s): "
        + ", ".join(sorted(_MISSING_COVERAGE_META))
    )


# ---- Per-organism configuration -----------------------------------------
# A categorical panel spec maps each present category to a hex colour. The
# optional "order" reorders the categories (and therefore the legend); the
# optional "halo_size" overrides MARKER_SIZE for the black outline layer.
DATASETS = {
    "dicty": {
        "data": ROOT / "data" / "dicty.h5ad",
        "time_key": "time",          # categorical "00hr".."20hr" -> numeric hours
        # 1D pseudotime line plot: which numeric time to average per bin, and the
        # x axis. time_col=None derives numeric hours from time_key.
        "line_plot": {
            "time_col": None,
            "xticks": [0, 4, 8, 12, 16, 20],
            "xlim": [0, 20],
            "tick_labels": None,
        },
        "categorical_panels": {
            "cell_type": {
                "obs_key": "cell_type",
                "colors": {"prespore": "#f46d43", "prestalk": "#3882b9"},
                "shuffle": True,
                "halo_size": 65,
                "legend": False,
            },
            "cAMP_genes": {
                "obs_key": "cAMP_genes",
                "colors": {"cAMP_genes": "#66c2a5"},
                "shuffle": False,
                "legend": False,
            },
            "milestones": {
                "obs_key": "milestones",
                "order": [
                    "noagg_ripple_up",
                    "ripple_lag_up",
                    "lag_tag_up",
                    "tag_tip_up",
                    "tip_slug_up",
                    "slug_Mhat_up",
                    "Mhat_cul_up",
                ],
                "colors": {
                    "noagg_ripple_up": "#9e0142",
                    "ripple_lag_up": "#f36c44",
                    "lag_tag_up": "#fedf8b",
                    "tag_tip_up": "#e6f799",
                    "tip_slug_up": "#66c2a4",
                    "slug_Mhat_up": "#5e51a2",
                    "Mhat_cul_up": "#1250b2",
                },
                "shuffle": False,
                "legend": True,
            },
        },
        # Grouped avgBIO bar chart: which gene-set JSONs to read and their labels.
        "bioconservation_gene_sets": {
            "cell_type": "Cell type",
            "cAMP_genes": "cAMP induced genes",
            "milestones": "Milestones",
        },
    },
    "elegans": {
        "data": ROOT / "data" / "elegans.h5ad",
        "time_key": "embryo.time",   # numeric minutes; coloured directly
        # Line plot averages the cleaned "time_numeric" column per bin; x axis in
        # minutes, tick labels relabelled to hours.
        "line_plot": {
            "time_col": "time_numeric",
            "xticks": [0, 162, 324, 486, 648, 810],
            "xlim": [0, 810],
            "tick_labels": ["0", "2.7", "5.4", "8.1", "10.8", "13.5"],
        },
        # Categorical panels use positional colour lists (one colour per category
        # in cat.categories order); no special highlighting.
        "categorical_panels": {
            "cell_type": {
                "obs_key": "cell_type",
                "colors": [
                    "#4c72b0", "#da8dc4", "#ffed6f", "#de8453", "#55a867",
                    "#c54e50", "#8073b2", "#ccb974", "#8cd3c7", "#ffffb2",
                ],
                "shuffle": False,
                "legend": True,
            },
            "developmental_markers": {
                "obs_key": "developmental_markers",
                "colors": ["#9e0142", "#fdae61", "#ffffbf", "#66c2a5"],
                "shuffle": False,
                "legend": True,
            },
            "lineage_markers": {
                "obs_key": "lineage_markers",
                "colors": ["#d98c5f", "#4f9fe6", "#9163e8", "#78b84d", "#deeb3b", "#ea459f"],
                "shuffle": False,
                "legend": True,
            },
        },
        "bioconservation_gene_sets": {
            "cell_type": "Cell type",
            "lineage_markers": "Lineages",
            "developmental_markers": "Developmental stages",
        },
    },
}


def output_root(base: Path | None = None) -> Path:
    """Directory under which ``<organism>/<panel>/<method>.svg`` is written."""
    return (base or ROOT / "manuscript" / "plots")


def embeddings_dir() -> Path:
    """Directory holding the per-method embedding manifests (gene counts)."""
    return ROOT / "results" / "embeddings"


def avgbio_dir(organism: str) -> Path:
    return ROOT / "results" / f"biological_benchmarks_{organism}"


def pseudotime_dir(organism: str) -> Path:
    return ROOT / "results" / f"pseudotime_{organism}"


def selected_results_file(base_dir: Path, filename: str) -> Path | None:
    """Resolve ``filename`` through the run pinned by ``xspecies select``.

    Pinned run (``runs/CURRENT``) -> legacy flat file -> ``None``. The figures use
    whichever run you explicitly pinned, not whatever ran most recently.
    """
    from select_run import current_file  # code/ is already on sys.path

    return current_file(base_dir, filename)
