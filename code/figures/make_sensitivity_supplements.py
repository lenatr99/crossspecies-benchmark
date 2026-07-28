"""Generate UCE-depth and PCA-preprocessing sensitivity supplements.

The caller supplies the two new pseudotime run identifiers explicitly. Main
result identifiers remain recorded defaults and are never changed by this
script. Outputs include LaTeX plus long-form TSVs with full-precision values.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "supplements"
ORGANISMS = ("dicty", "elegans")
PANELS = {
    "dicty": (
        ("cell_type", "Cell type"),
        ("cAMP_genes", "cAMP-pulse-induced genes"),
        ("milestones", "Milestones"),
    ),
    "elegans": (
        ("cell_type", "Cell/tissue classes"),
        ("lineage_markers", "Lineage categories"),
        ("developmental_markers", "Developmental markers"),
    ),
}
ORG_TEX = {"dicty": r"\species{D.~discoideum}", "elegans": r"\species{C.~elegans}"}
DIMENSION_CONDITIONS = (
    ("PCA", "X_pca", "X_pca"),
    ("scVI", "X_scVI", "X_scVI"),
    ("scGPT", "X_scGPT_ortholog_tier12", "X_scGPT_ortholog_tier12_pca100_supp"),
    ("Geneformer", "X_geneformer_ortholog_tier12", "X_geneformer_ortholog_tier12_pca100_supp"),
    ("UCE-4L", "X_uce_4L", "X_uce_4L_pca100_supp"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avgbio-main-run", default="20260529_124606")
    parser.add_argument("--avgbio-pca-run", default="20260722_091301")
    parser.add_argument("--pseudotime-main-point-run", default="20260703_130811")
    parser.add_argument("--pseudotime-main-stability-run", default="20260709_222558")
    parser.add_argument("--pseudotime-sensitivity-point-run", required=True)
    parser.add_argument("--pseudotime-sensitivity-stability-run", required=True)
    parser.add_argument("--avgbio-dimension-run", required=True)
    parser.add_argument("--pseudotime-expanded-point-run", required=True)
    parser.add_argument("--pseudotime-expanded-stability-run", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.load(open(path))


def avgbio(org: str, run: str, panel: str, key: str) -> float:
    path = ROOT / f"results/biological_benchmarks_{org}/runs/{run}/{panel}.json"
    return float(load_json(path)[key]["avg_bio"])


def pseudotime(org: str, run: str, key: str) -> dict:
    path = ROOT / f"results/pseudotime_{org}/runs/{run}/pseudotime.json"
    return load_json(path)[key]


def dimension_variance(org: str, source_key: str) -> float:
    if source_key in {"X_pca", "X_scVI"}:
        return 1.0
    manifest = load_json(
        ROOT / "results" / "embeddings" / f"dimension_matched_100_{org}_manifest.json"
    )
    matches = [
        record for record in manifest["records"] if record["source_key"] == source_key
    ]
    if len(matches) != 1:
        raise AssertionError((org, source_key, len(matches)))
    return float(matches[0]["explained_variance_ratio_sum"])


def scale(value: float) -> float:
    return (float(value) + 1.0) / 2.0


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def interval(record: dict, metric: str) -> tuple[float, float]:
    low, high = record[f"{metric}_ci"]
    return scale(low), scale(high)


def write_tsv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def make_s4(args: argparse.Namespace) -> None:
    conditions = (
        ("UCE-4L", "X_uce_4L"),
        ("UCE-33L", "X_uce_33L"),
    )
    rows = []
    table_rows = []
    for org in ORGANISMS:
        for panel, panel_label in PANELS[org]:
            values = [avgbio(org, args.avgbio_main_run, panel, key) for _, key in conditions]
            table_rows.append((org, panel_label, values))
            rows.extend((org, panel, label, value) for (label, _), value in zip(conditions, values))
    write_tsv(args.output_dir / "S4_uce_layer_calibration.tsv", ["organism", "annotation_panel", "condition", "avgBIO"], rows)
    body = []
    last_org = None
    for org, panel_label, values in table_rows:
        org_cell = ORG_TEX[org] if org != last_org else ""
        body.append(" & ".join([org_cell, panel_label, *[fmt(v) for v in values]]) + r" \\")
        last_org = org
        if panel_label == "Milestones":
            body.append(r"\addlinespace")
    pseudo_rows = []
    pseudo_body = []
    for org in ORGANISMS:
        records = []
        for label, key, point_run, stability_run in (
            ("UCE-4L", "X_uce_4L", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
            ("UCE-33L", "X_uce_33L", args.pseudotime_sensitivity_point_run, args.pseudotime_sensitivity_stability_run),
        ):
            point = pseudotime(org, point_run, key)
            stability = pseudotime(org, stability_run, key)
            values = []
            for metric in ("spearman", "lins_ccc_consensus"):
                estimate = scale(point[metric])
                low, high = interval(stability, metric)
                pseudo_rows.append((org, label, metric, estimate, low, high, point_run, stability_run))
                values.append((estimate, low, high))
            records.append(values)
        for metric_index, metric_label in enumerate((r"Spearman $\rho$", "Consensus Lin's CCC")):
            cells = [
                f"{fmt(values[metric_index][0])} [{fmt(values[metric_index][1])}, {fmt(values[metric_index][2])}]"
                for values in records
            ]
            pseudo_body.append(
                " & ".join([ORG_TEX[org] if metric_index == 0 else "", metric_label, *cells]) + r" \\"
            )
        if org == "dicty":
            pseudo_body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S5_uce_pseudotime.tsv",
        ["organism", "condition", "metric", "point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled", "point_run", "stability_run"],
        pseudo_rows,
    )

    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{UCE checkpoint-depth sensitivity.} The upper block reports full-data avgBIO point estimates. The lower block reports full-data developmental-ordering estimates with 2.5th--97.5th percentile stability intervals from 30 repeated 90\% subsamples. Correlations are displayed on $[0,1]$ via $(r+1)/2$.}
\label{tab:s4}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & Annotation & UCE-4L & UCE-33L \\
\midrule
""" + "\n".join(body) + r"""
\midrule
Organism & Metric & UCE-4L [interval] & UCE-33L [interval] \\
\midrule
""" + "\n".join(pseudo_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (args.output_dir / "S4_uce_layer_calibration.tex").write_text(tex, encoding="utf-8")


def make_s5(args: argparse.Namespace) -> None:
    pseudo_rows = []
    pseudo_body = []
    for org in ORGANISMS:
        for label, key, point_run, stability_run in (
            ("UCE-4L", "X_uce_4L", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
            ("UCE-33L", "X_uce_33L", args.pseudotime_sensitivity_point_run, args.pseudotime_sensitivity_stability_run),
        ):
            point = pseudotime(org, point_run, key)
            stability = pseudotime(org, stability_run, key)
            values = []
            for metric in ("spearman", "lins_ccc_consensus"):
                estimate = scale(point[metric])
                low, high = interval(stability, metric)
                values.extend((estimate, low, high))
                pseudo_rows.append((org, label, metric, estimate, low, high, point_run, stability_run))
            pseudo_body.append(
                f"{ORG_TEX[org] if label == 'UCE-4L' else ''} & {label} & "
                f"{fmt(values[0])} [{fmt(values[1])}, {fmt(values[2])}] & "
                f"{fmt(values[3])} [{fmt(values[4])}, {fmt(values[5])}] \\\\"
            )
        if org == "dicty":
            pseudo_body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S5_uce_pseudotime.tsv",
        ["organism", "condition", "metric", "point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled", "point_run", "stability_run"],
        pseudo_rows,
    )

    diag_rows = []
    diag_body = []
    fit_count_rows = []
    shared_human_count = None
    for org in ORGANISMS:
        adapter_data = {}
        for adapter_slug in ("whitening", "macrogene"):
            base = ROOT / f"results/embeddings/uce/esm2_adapter/{org}_{adapter_slug}_global"
            manifest = load_json(base / f"{org}_{adapter_slug}_global_manifest.json")
            with open(manifest["pair_diagnostics"], encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            tier_counts = {
                tier: sum(row["tier"] == str(tier) for row in pair_rows)
                for tier in (1, 2, 3)
            }
            adapter_data[adapter_slug] = (manifest, tier_counts)

        whitening_manifest, whitening_tiers = adapter_data["whitening"]
        macrogene_manifest, macrogene_tiers = adapter_data["macrogene"]
        assert whitening_tiers == macrogene_tiers
        assert int(whitening_manifest["n_source_genes"]) == int(macrogene_manifest["n_source_genes"])
        assert int(whitening_manifest["n_human_genes"]) == int(macrogene_manifest["n_human_genes"])
        if shared_human_count is None:
            shared_human_count = int(whitening_manifest["n_human_genes"])
            fit_count_rows.append(("global_fit_human", "all", shared_human_count))
        else:
            assert shared_human_count == int(whitening_manifest["n_human_genes"])
        fit_count_rows.extend(
            (
                ("global_fit_target", org, int(whitening_manifest["n_source_genes"])),
                (
                    "macrogene_anchor_pairs",
                    org,
                    int(macrogene_manifest["method_parameters"]["n_anchor_pairs"]),
                ),
            )
        )

        records = [("Nearest human", "mean_nn_cosine", int(whitening_manifest["n_source_genes"]))]
        records.extend(
            (f"Assigned ortholog T{tier}", f"tier{tier}_assigned_cosine", whitening_tiers[tier])
            for tier in (1, 2, 3)
        )
        for record_index, (label, prefix, n_eval) in enumerate(records):
            before = float(whitening_manifest["metrics"][f"{prefix}_before"])
            macrogene_before = float(macrogene_manifest["metrics"][f"{prefix}_before"])
            assert abs(before - macrogene_before) < 1e-12
            whitening_after = float(whitening_manifest["metrics"][f"{prefix}_after"])
            macrogene_after = float(macrogene_manifest["metrics"][f"{prefix}_after"])
            diag_rows.append((org, label, before, whitening_after, macrogene_after, n_eval))
            diag_body.append(
                f"{ORG_TEX[org] if record_index == 0 else ''} & {label} & {fmt(before)} & "
                f"{fmt(whitening_after)} & {fmt(macrogene_after)} & {n_eval:,} \\\\"
            )
        if org == "dicty":
            diag_body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S5_protein_diagnostics.tsv",
        [
            "organism",
            "diagnostic",
            "cosine_before",
            "cosine_after_global_whitening",
            "cosine_after_macrogene_residual",
            "n_evaluation_records",
        ],
        diag_rows,
    )
    write_tsv(
        args.output_dir / "S5_protein_calibration_counts.tsv",
        ["component", "organism", "n_records"],
        fit_count_rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{UCE checkpoint-depth pseudotime sensitivity.} Full-data point estimates and 2.5th--97.5th percentile stability intervals from 30 repeated 90\% subsamples. Both correlations are displayed on $[0,1]$ via $(r+1)/2$.}
\label{tab:s5}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & Checkpoint & Spearman $\rho$ [interval] & Consensus Lin's CCC [interval] \\
\midrule
""" + "\n".join(pseudo_body) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{\textbf{Protein-space diagnostics before and after cell-label-free calibration.} ``Before'' denotes raw ESM-2 vectors. Global moment matching used all 13{,}209 \species{D.~discoideum} or 19{,}985 \species{C.~elegans} target proteins and the shared dictionary of 19{,}790 human proteins. The macrogene adapter began with that transform and used 960 or 1{,}638 Tier~1 anchor pairs, respectively. Nearest-human cosine is evaluated for all target proteins; assigned-ortholog cosine uses the recorded consensus pairs in each tier. The calibrated columns report global matching alone and the complete macrogene-residual adapter on the same evaluation sets.}
\label{tab:s6}
\small
\begin{tabular}{@{}llcccc@{}}
\toprule
Organism & Diagnostic & Before & Global matching & Macrogene residual & Evaluation $n$ \\
\midrule
""" + "\n".join(diag_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (args.output_dir / "S5_uce_pseudotime_protein.tex").write_text(tex, encoding="utf-8")


def make_s6(args: argparse.Namespace) -> None:
    avg_rows = []
    avg_body = []
    for org in ORGANISMS:
        for panel, panel_label in PANELS[org]:
            raw = avgbio(org, args.avgbio_main_run, panel, "X_pca")
            processed = avgbio(org, args.avgbio_pca_run, panel, "X_pca_hvg_scaled")
            delta = processed - raw
            avg_rows.append((org, panel, raw, processed, delta))
            avg_body.append(
                f"{ORG_TEX[org] if panel == 'cell_type' else ''} & {panel_label} & "
                f"{fmt(raw)} & {fmt(processed)} & {fmt(delta, 3)} \\\\"
            )
        if org == "dicty":
            avg_body.append(r"\addlinespace")

    pseudo_rows = []
    pseudo_body = []
    for org in ORGANISMS:
        for label, key, point_run, stability_run in (
            ("Raw-count PCA", "X_pca", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
            ("Preprocessed PCA", "X_pca_hvg_scaled", args.pseudotime_sensitivity_point_run, args.pseudotime_sensitivity_stability_run),
        ):
            point = pseudotime(org, point_run, key)
            stability = pseudotime(org, stability_run, key)
            row_values = []
            for metric in ("spearman", "lins_ccc_consensus"):
                estimate = scale(point[metric])
                low, high = interval(stability, metric)
                pseudo_rows.append((org, label, metric, estimate, low, high, point_run, stability_run))
                row_values.extend((estimate, low, high))
            pseudo_body.append(
                f"{ORG_TEX[org] if label == 'Raw-count PCA' else ''} & {label} & "
                f"{fmt(row_values[0])} [{fmt(row_values[1])}, {fmt(row_values[2])}] & "
                f"{fmt(row_values[3])} [{fmt(row_values[4])}, {fmt(row_values[5])}] \\\\"
            )
        if org == "dicty":
            pseudo_body.append(r"\addlinespace")
    write_tsv(args.output_dir / "S6_pca_avgbio.tsv", ["organism", "annotation_panel", "raw_PCA_avgBIO", "preprocessed_PCA_avgBIO", "delta_preprocessed_minus_raw"], avg_rows)
    write_tsv(
        args.output_dir / "S6_pca_pseudotime.tsv",
        ["organism", "condition", "metric", "point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled", "point_run", "stability_run"],
        pseudo_rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{PCA preprocessing sensitivity for annotation preservation.} Raw-count PCA is the minimal headline baseline. The sensitivity recipe selects 2,000 Seurat-v3 highly variable genes, normalizes to 10,000 counts per cell, applies \texttt{log1p}, scales with a cap of 10 and computes 100 PCs. Delta is preprocessed minus raw.}
\label{tab:s5}
\small
\begin{tabular}{@{}llrrr@{}}
\toprule
Organism & Annotation & Raw PCA & Preprocessed PCA & Delta \\
\midrule
""" + "\n".join(avg_body) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{\textbf{PCA preprocessing sensitivity for developmental ordering.} Full-data point estimates and 2.5th--97.5th percentile stability intervals from 30 repeated 90\% subsamples; correlations are displayed on $[0,1]$ via $(r+1)/2$.}
\label{tab:s6}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & PCA condition & Spearman $\rho$ [interval] & Consensus Lin's CCC [interval] \\
\midrule
""" + "\n".join(pseudo_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (args.output_dir / "S6_pca_preprocessing.tex").write_text(tex, encoding="utf-8")


def make_headline_pseudotime_table(args: argparse.Namespace) -> None:
    """Complete headline pseudotime table using consensus CCC throughout."""
    conditions = (
        ("PCA", "X_pca"),
        ("scVI", "X_scVI"),
        ("Geneformer", "X_geneformer_ortholog_tier12"),
        ("scGPT", "X_scGPT_ortholog_tier12"),
        ("UCE-4L", "X_uce_4L"),
    )
    rows, body = [], []
    for org in ORGANISMS:
        for index, (label, key) in enumerate(conditions):
            point = pseudotime(org, args.pseudotime_main_point_run, key)
            stability = pseudotime(org, args.pseudotime_main_stability_run, key)
            values = []
            for metric in ("spearman", "lins_ccc_consensus"):
                estimate = scale(point[metric])
                low, high = interval(stability, metric)
                rows.append((org, label, metric, estimate, low, high))
                values.extend((estimate, low, high))
            body.append(
                f"{ORG_TEX[org] if index == 0 else ''} & {label} & "
                f"{fmt(values[0])} [{fmt(values[1])}, {fmt(values[2])}] & "
                f"{fmt(values[3])} [{fmt(values[4])}, {fmt(values[5])}] \\\\"
            )
        if org == "dicty":
            body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S6_pseudotime_stability.tsv",
        ["organism", "method", "metric", "point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled"],
        rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{Pseudotime point estimates and repeated-subsampling stability intervals.} Point estimates come from the frozen full-data outputs. Brackets show the 2.5th--97.5th percentiles across 30 paired repeats from the completed stability analysis, each sampling 90\% of cells without replacement and recomputing the neighbour graph and DPT. Spearman is averaged across roots; consensus Lin's CCC is computed on the root-averaged cell-level pseudotime. All correlations are displayed on $[0,1]$ via $(r+1)/2$. These are stability intervals, not population confidence intervals.}
\label{tab:s8}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & Method & Spearman $\rho$ [stability interval] & Consensus Lin's CCC [stability interval] \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
    (args.output_dir / "S6_pseudotime_stability.tex").write_text(tex, encoding="utf-8")


def make_s9(args: argparse.Namespace) -> None:
    """Coverage-matched UCE-4L developmental-ordering sensitivity."""
    conditions = (
        ("Full eligible gene set", "X_uce_4L", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
        ("Ortholog-available gene set", "X_uce_4L_orthoT12", args.pseudotime_expanded_point_run, args.pseudotime_expanded_stability_run),
    )
    rows, body = [], []
    for org in ORGANISMS:
        for label, key, point_run, stability_run in conditions:
            point = pseudotime(org, point_run, key)
            stability = pseudotime(org, stability_run, key)
            values = []
            for metric in ("spearman", "lins_ccc_consensus"):
                estimate = scale(point[metric])
                low, high = interval(stability, metric)
                rows.append((org, label, metric, estimate, low, high, point_run, stability_run))
                values.extend((estimate, low, high))
            body.append(
                f"{ORG_TEX[org] if label == 'Full eligible gene set' else ''} & {label} & "
                f"{fmt(values[0])} [{fmt(values[1])}, {fmt(values[2])}] & "
                f"{fmt(values[3])} [{fmt(values[4])}, {fmt(values[5])}] \\\\"
            )
        if org == "dicty":
            body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S9_coverage_matched_pseudotime.tsv",
        ["organism", "condition", "metric", "point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled", "point_run", "stability_run"],
        rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{Coverage-matched UCE-4L sensitivity for developmental ordering.} The full UCE-4L representation uses all eligible target-species proteins; the matched condition is reconstructed from genes with Tier~1--2 source-to-human ortholog assignments, matching the ortholog-available input pool used by fixed-vocabulary models. Values are full-data point estimates with 2.5th--97.5th percentile intervals from 30 repeated 90\% subsamples. Correlations are displayed on $[0,1]$ via $(r+1)/2$.}
\label{tab:s9}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & UCE-4L gene set & Spearman $\rho$ [interval] & Consensus Lin's CCC [interval] \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (args.output_dir / "S9_coverage_matched_pseudotime.tex").write_text(tex, encoding="utf-8")


def make_s10(args: argparse.Namespace) -> None:
    """Uniform post-embedding 100-component sensitivity for avgBIO."""
    rows, body = [], []
    for org in ORGANISMS:
        for panel, panel_label in PANELS[org]:
            cells = []
            for method, native_key, reduced_key in DIMENSION_CONDITIONS:
                native = avgbio(org, args.avgbio_main_run, panel, native_key)
                reduced_run = (
                    args.avgbio_main_run
                    if native_key == reduced_key
                    else args.avgbio_dimension_run
                )
                reduced = avgbio(org, reduced_run, panel, reduced_key)
                delta = reduced - native
                rows.append((org, panel, method, native_key, reduced_key, native, reduced, delta))
                display_delta = 0.0 if abs(delta) < 0.0005 else delta
                cells.append(
                    f"{fmt(native)} $\\to$ {fmt(reduced)} ({display_delta:+.3f})"
                )
            body.append(
                " & ".join([ORG_TEX[org] if panel == "cell_type" else "", panel_label, *cells]) + r" \\"
            )
        if org == "dicty":
            body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S10_dimension100_avgbio.tsv",
        ["organism", "annotation_panel", "method", "native_key", "dimension100_key", "native_avgBIO", "dimension100_avgBIO", "delta_dimension100_minus_native"],
        rows,
    )
    variance_rows, variance_body = [], []
    for org in ORGANISMS:
        values = []
        for method, native_key, _ in DIMENSION_CONDITIONS:
            retained = dimension_variance(org, native_key)
            variance_rows.append((org, method, native_key, retained))
            values.append(fmt(retained))
        variance_body.append(" & ".join([ORG_TEX[org], *values]) + r" \\")
    write_tsv(
        args.output_dir / "S10_dimension100_variance.tsv",
        ["organism", "method", "native_key", "explained_variance_ratio_retained"],
        variance_rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{Common 100-component sensitivity for annotation preservation.} The native PCA and scVI representations are already 100-dimensional and are reused unchanged; scGPT, Geneformer and UCE-4L are independently reduced to 100 components by centered, unscaled PCA. Cells show native $\to$ 100-dimensional avgBIO (change). This analysis tests sensitivity to unequal representation dimensionality; it does not place the methods in a shared coordinate system.}
\label{tab:s10}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccc@{}}
\toprule
Organism & Annotation & PCA & scVI & scGPT & Geneformer & UCE-4L \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\medskip

\begin{tabular}{@{}lccccc@{}}
\toprule
Organism & PCA & scVI & scGPT & Geneformer & UCE-4L \\
\midrule
""" + "\n".join(variance_body) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\begin{minipage}{0.96\textwidth}\scriptsize Values in the lower block are the fractions of variance retained by the 100 components.\end{minipage}
\end{table}
"""
    (args.output_dir / "S10_dimension100_avgbio.tex").write_text(tex, encoding="utf-8")


def make_s11(args: argparse.Namespace) -> None:
    """Uniform post-embedding 100-component sensitivity for pseudotime."""
    rows, body = [], []
    for org in ORGANISMS:
        for method, native_key, reduced_key in DIMENSION_CONDITIONS:
            native = pseudotime(org, args.pseudotime_main_point_run, native_key)
            reduced_point_run = (
                args.pseudotime_main_point_run
                if native_key == reduced_key
                else args.pseudotime_expanded_point_run
            )
            reduced_stability_run = (
                args.pseudotime_main_stability_run
                if native_key == reduced_key
                else args.pseudotime_expanded_stability_run
            )
            reduced = pseudotime(org, reduced_point_run, reduced_key)
            stability = pseudotime(org, reduced_stability_run, reduced_key)
            cells = []
            for metric in ("spearman", "lins_ccc_consensus"):
                native_value = scale(native[metric])
                reduced_value = scale(reduced[metric])
                low, high = interval(stability, metric)
                rows.append((org, method, metric, native_value, reduced_value, reduced_value - native_value, low, high, reduced_point_run, reduced_stability_run))
                cells.append(f"{fmt(native_value)} $\\to$ {fmt(reduced_value)} [{fmt(low)}, {fmt(high)}]")
            body.append(
                f"{ORG_TEX[org] if method == 'PCA' else ''} & {method} & {cells[0]} & {cells[1]} \\\\"
            )
        if org == "dicty":
            body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S11_dimension100_pseudotime.tsv",
        ["organism", "method", "metric", "native_point_rescaled", "dimension100_point_rescaled", "delta_dimension100_minus_native", "dimension100_stability_low_rescaled", "dimension100_stability_high_rescaled", "point_run", "stability_run"],
        rows,
    )
    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{Common 100-component sensitivity for developmental ordering.} PCA and scVI are already 100-dimensional and are reused unchanged; the other three representations are reduced by post-embedding PCA. Each cell shows native $\to$ 100-dimensional full-data point estimate, followed by the 2.5th--97.5th percentile stability interval for that 100-dimensional condition from 30 repeated 90\% subsamples. Correlations are displayed on $[0,1]$ via $(r+1)/2$.}
\label{tab:s11}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llcc@{}}
\toprule
Organism & Method & Spearman $\rho$: native $\to$ 100D [interval] & Consensus Lin's CCC: native $\to$ 100D [interval] \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
    (args.output_dir / "S11_dimension100_pseudotime.tex").write_text(tex, encoding="utf-8")


def make_s12(args: argparse.Namespace) -> None:
    """Cell-level and protein-level diagnostics for UCE protein calibration."""
    conditions = (
        ("UCE-4L", "X_uce_4L"),
        ("UCE-4L global", "X_uce_4L_whitening"),
        ("UCE-33L", "X_uce_33L"),
        ("UCE-33L global", "X_uce_33L_whitening"),
        ("UCE-33L macrogene", "X_uce_33L_macrogene"),
    )
    rows, body = [], []
    for org in ORGANISMS:
        for panel, panel_label in PANELS[org]:
            values = [
                avgbio(org, args.avgbio_main_run, panel, key)
                for _, key in conditions
            ]
            rows.extend(
                (org, panel, label, key, value)
                for (label, key), value in zip(conditions, values)
            )
            body.append(
                f"{ORG_TEX[org] if panel == 'cell_type' else ''} & {panel_label} & "
                + " & ".join(fmt(value) for value in values)
                + r" \\"
            )
        if org == "dicty":
            body.append(r"\addlinespace")
    write_tsv(
        args.output_dir / "S12_uce_protein_calibration.tsv",
        ["organism", "annotation_panel", "condition", "embedding_key", "avgBIO"],
        rows,
    )

    with (args.output_dir / "S5_protein_diagnostics.tsv").open(encoding="utf-8") as handle:
        diagnostics = list(csv.DictReader(handle, delimiter="\t"))
    diagnostic_body = []
    for index, row in enumerate(diagnostics):
        diagnostic_body.append(
            f"{ORG_TEX[row['organism']] if row['diagnostic'] == 'Nearest human' else ''} & "
            f"{row['diagnostic']} & {fmt(float(row['cosine_before']))} & "
            f"{fmt(float(row['cosine_after_global_whitening']))} & "
            f"{fmt(float(row['cosine_after_macrogene_residual']))} & "
            f"{int(row['n_evaluation_records']):,} \\\\"
        )
        if index == 3:
            diagnostic_body.append(r"\addlinespace")

    tex = r"""\begin{table}[H]
\centering
\caption{\textbf{UCE protein-space calibration sensitivity.} The upper block reports full-data avgBIO before and after two cell-label-free protein-space adapters. ``Global'' denotes global moment matching; ``macrogene'' denotes global matching followed by a residual transform estimated from Tier~1 anchor pairs. The lower block reports mean nearest-human and assigned-ortholog cosine values for the same raw and adapted protein vectors. Global matching used 13{,}209 \species{D.~discoideum} or 19{,}985 \species{C.~elegans} target proteins and one shared set of 19{,}790 human proteins; the macrogene adapter used 960 or 1{,}638 Tier~1 anchor pairs, respectively.}
\label{tab:s12}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccc@{}}
\toprule
Organism & Annotation & 4L raw & 4L global & 33L raw & 33L global & 33L macrogene \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\medskip

\begin{tabular}{@{}llcccc@{}}
\toprule
Organism & Protein diagnostic & Raw & Global & Macrogene & Evaluation $n$ \\
\midrule
""" + "\n".join(diagnostic_body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (args.output_dir / "S12_uce_protein_calibration.tex").write_text(
        tex, encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_s4(args)
    make_s5(args)
    make_s6(args)
    make_headline_pseudotime_table(args)
    make_s9(args)
    make_s10(args)
    make_s11(args)
    make_s12(args)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "evaluation_parameters": {
            "processed_pca_avgbio": {
                "seed": 0,
                "n_jobs": 1,
                "neighbor_metric": "euclidean",
                "subsample_repeats": 0,
            },
            "sensitivity_pseudotime_point": {
                "inference": "bootstrap",
                "seed": 0,
                "n_candidates": 500,
                "root_method": "uniform",
                "k_roots": 50,
                "n_boot": 1000,
                "alpha": 0.05,
                "bootstrap_consensus_roots": True,
            },
            "sensitivity_pseudotime_stability": {
                "inference": "subsample",
                "seed": 0,
                "n_candidates": 500,
                "n_repeats": 30,
                "subsample_frac": 0.9,
                "k_roots": 20,
                "alpha": 0.05,
            },
            "dimension_matched_embeddings": {
                "operation": "independent PCA for frozen embeddings above 100D; native 100D PCA and scVI reused",
                "n_components": 100,
                "centered": True,
                "scaled": False,
                "svd_solver": "randomized",
                "seed": 42,
            },
        },
        "outputs": sorted(str(path) for path in args.output_dir.glob("S*.tex"))
        + sorted(str(path) for path in args.output_dir.glob("S*.tsv")),
    }
    path = args.output_dir / "sensitivity_supplements_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
