#!/usr/bin/env python3
"""Assemble publication figures from Illustrator masters and rendered panels.

This is the second stage of figure generation:

    xspecies figures --dataset dicty
    xspecies assemble-figures --dataset dicty

The first command writes data-driven SVG panels to ``manuscript/plots``. This
command replaces the placeholders in the tracked Illustrator master, writes a
composed SVG, and exports PDF/PNG publication assets with Inkscape.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from split_panels import process as split_figure
from svg_panel_replace import (
    FLIP_MODELS,
    FLIP_PCA_UMAP,
    FLIP_ROT180,
    apply_groups,
    find_inkscape,
    replace_group,
    strip_illustrator_pgf,
)

DATASETS = ("dicty", "elegans")
GRAPHIK = "GraphikTrial-Regular, 'Graphik Trial'"
GRAPHIK_SEMIBOLD = "GraphikCompactTrial-Semibold, 'Graphik Compact Trial'"
ARIAL_TO_GRAPHIK = {"Arial": GRAPHIK}
ARIAL_BOLD_TO_GRAPHIK = {"Arial": GRAPHIK_SEMIBOLD}

PANEL_FILES = {
    "pca": "X_pca.svg",
    "scvi": "X_scVI.svg",
    "uce": "X_uce_4L.svg",
    "scgpt": "X_scGPT_ortholog_tier12.svg",
    "geneformer": "X_geneformer_ortholog_tier12.svg",
}
LINEPLOT_FILES = {
    "lineplot-pca": "1dplot_pseudotime_x_pca.svg",
    "lineplot-scvi": "1dplot_pseudotime_x_scvi.svg",
    "lineplot-uce": "1dplot_pseudotime_x_uce_4l.svg",
    "lineplot-scgpt": "1dplot_pseudotime_x_scgpt_ortholog_tier12.svg",
    "lineplot-geneformer": "1dplot_pseudotime_x_geneformer_ortholog_tier12.svg",
}
CONFIG = {
    "dicty": {
        # The original Illustrator group's glyph-aware bounds extend beyond the
        # artboard. Constrain the replacement slot so right-side coverage labels
        # remain inside the exported page rather than being clipped.
        "coverage_box": (926.03, 549.19, 760.0, 513.31),
        "groups": {
            "time": "time",
            "pseudotime": "pseudotime",
            "cell-type": "cell_type",
            "camp-genes": "cAMP_genes",
            "milestones": "milestones",
        },
        "flips": {
            "pca": FLIP_PCA_UMAP,
            "scvi": FLIP_ROT180,
            "uce": FLIP_MODELS,
            "scgpt": FLIP_MODELS,
            "geneformer": FLIP_MODELS,
        },
    },
    "elegans": {
        "coverage_box": (929.89, 569.74, 770.0, 510.65),
        "groups": {
            "time": "time",
            "pseudotime": "pseudotime",
            "cell-type": "cell_type",
            "developmental-stages": "developmental_markers",
            "lineage": "lineage_markers",
        },
        "flips": {
            "pca": "hr270",
            "scvi": "vr90",
            "uce": "v",
            "scgpt": "v",
            "geneformer": "v",
        },
    },
}


def paths_for(
    organism: str, plots_root: Path, sources_root: Path, output_root: Path
) -> tuple[Path, Path, list[tuple[Path, dict]], dict]:
    """Build all source and replacement paths for one organism."""
    config = CONFIG[organism]
    master = sources_root / f"{organism}_1.svg"
    output = output_root / f"{organism}_1.svg"
    organism_plots = plots_root / organism

    groups = []
    for element_prefix, plot_dir in config["groups"].items():
        jobs = {
            f"{element_prefix}-{method}": (
                filename,
                config["flips"][method],
            )
            for method, filename in PANEL_FILES.items()
        }
        groups.append((organism_plots / plot_dir, jobs))

    group_jobs = {
        "d-panel": {
            "svg_path": organism_plots
            / "genome_coverage"
            / f"genome_coverage_{organism}.svg",
            "layer": "axes_1",
            "box": config["coverage_box"],
            "fonts": ARIAL_TO_GRAPHIK,
            "bold_fonts": ARIAL_BOLD_TO_GRAPHIK,
        },
        "f-panel": {
            "svg_path": organism_plots / "pseudotime" / "pseudotime_metrics.svg",
            "layer": "axes_1",
            "fonts": ARIAL_TO_GRAPHIK,
        },
        "h-panel": {
            "svg_path": organism_plots
            / "bioconservation"
            / "bioconservation_metrics.svg",
            "layer": "axes_1",
            "fonts": ARIAL_TO_GRAPHIK,
        },
    }
    lineplot_dir = organism_plots / "line_plot"
    for element_id, filename in LINEPLOT_FILES.items():
        group_jobs[element_id] = {
            "svg_path": lineplot_dir / filename,
            "layer": "axes_1",
            "fonts": ARIAL_TO_GRAPHIK,
        }
    return master, output, groups, group_jobs


def required_inputs(master: Path, groups: list[tuple[Path, dict]], group_jobs: dict) -> list[Path]:
    """Return every input path used by an assembly job."""
    paths = [master]
    for svg_dir, jobs in groups:
        paths.extend(svg_dir / filename for filename, _flip in jobs.values())
    paths.extend(Path(job["svg_path"]) for job in group_jobs.values())
    return paths


def validate_inputs(jobs: list[tuple[str, Path, Path, list, dict]]) -> None:
    missing = []
    for organism, master, _output, groups, group_jobs in jobs:
        missing.extend(
            f"{organism}: {path}"
            for path in required_inputs(master, groups, group_jobs)
            if not path.is_file()
        )
    if missing:
        details = "\n  ".join(missing)
        raise SystemExit(
            "assemble-figures: required inputs are missing:\n"
            f"  {details}\n"
            "Run `xspecies figures --dataset <dataset>` first and ensure its "
            "benchmark runs are pinned with `xspecies select`."
        )


def warn_for_fonts() -> None:
    """Warn when fontconfig can prove the publication fonts are unavailable."""
    fc_match = shutil.which("fc-match")
    if not fc_match:
        return
    missing = []
    for family in ("Graphik Trial", "Graphik Compact Trial"):
        result = subprocess.run(
            [fc_match, "-f", "%{family}", family],
            check=False,
            capture_output=True,
            text=True,
        )
        if family.lower() not in result.stdout.lower():
            missing.append(family)
    if missing:
        print(
            "warning: publication font(s) not found: "
            + ", ".join(missing)
            + "; Inkscape may substitute fonts and alter layout",
            file=sys.stderr,
        )


def export_figure(svg_path: Path, inkscape: str) -> None:
    """Export a composed SVG to manuscript PDF and 300-DPI PNG."""
    for extension in ("pdf", "png"):
        output = svg_path.with_suffix(f".{extension}")
        command = [
            inkscape,
            str(svg_path),
            "--export-area-page",
            f"--export-type={extension}",
            f"--export-filename={output}",
        ]
        if extension == "png":
            command.extend(
                [
                    "--export-dpi=300",
                    "--export-background=white",
                    "--export-background-opacity=1",
                ]
            )
        subprocess.run(command, check=True)
        print(f"wrote {output}")


def assemble_one(
    organism: str,
    master: Path,
    output: Path,
    groups: list,
    group_jobs: dict,
    *,
    inkscape: str,
    split_panels: bool,
) -> None:
    print(f"[{organism}] assembling {master} -> {output}", flush=True)
    source = apply_groups(master, None, groups)
    for element_id, kwargs in group_jobs.items():
        source = replace_group(source, element_id, inkscape=inkscape, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strip_illustrator_pgf(source))
    print(f"wrote {output}")
    export_figure(output, inkscape)
    if split_panels:
        split_figure(
            output,
            output.parent / "panels" / organism,
            inkscape=inkscape,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Dataset(s) to assemble (default: all).",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=ROOT / "manuscript" / "plots",
        help="Root containing rendered panel directories.",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=ROOT / "manuscript" / "figure_sources",
        help="Root containing tracked Illustrator master SVGs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "manuscript" / "figures",
        help="Destination for composed SVG/PDF/PNG files.",
    )
    parser.add_argument(
        "--split-panels",
        action="store_true",
        help="Also export each a-panel, b-panel, ... as SVG/PDF/PNG.",
    )
    parser.add_argument("--inkscape", help="Explicit path to the Inkscape executable.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all source files without composing or exporting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    organisms = DATASETS if args.dataset == "all" else (args.dataset,)
    jobs = [
        (organism, *paths_for(organism, args.plots_root, args.sources_root, args.output_root))
        for organism in organisms
    ]
    validate_inputs(jobs)
    if args.validate_only:
        print("All figure assembly inputs are present.")
        return 0

    inkscape = find_inkscape(args.inkscape)
    warn_for_fonts()
    for organism, master, output, groups, group_jobs in jobs:
        assemble_one(
            organism,
            master,
            output,
            groups,
            group_jobs,
            inkscape=inkscape,
            split_panels=args.split_panels,
        )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
