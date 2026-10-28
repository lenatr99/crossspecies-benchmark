"""Validate manuscript-facing sensitivity artifacts against their sources.

Run this after generating the sensitivity tables and protein-space figure.  It
does not modify data.  The PCA environment is required because the checks read
the AnnData objects and hash the stored arrays directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUPP = ROOT / "results" / "supplements"
RAW_HASHES = {
    ("dicty", "X_pca"): "9f7acff2546bba90660ddaba6d345dde6c481b5e86f6f436fd48c6ff7c36b042",
    ("dicty", "X_pca_50"): "34bbdd824332274ff5eca48c980c0836ee71524b6356210194e63efea696f410",
    ("elegans", "X_pca"): "a3c81f5eeb4e1fce36cc9e451a6410c62f4eacdbdfc6e635bd0cfbd826a141f1",
    ("elegans", "X_pca_50"): "827a88c41288c932bd5ecd6c1db538740778e87710ffb497116e28533d2b5313",
}
PROCESSED_HASHES = {
    ("dicty", "X_pca_hvg_scaled"): "071b679d0de84e30f15ed275f32476deb81fa11c46998bf05722e1f9edef5738",
    ("dicty", "X_pca_hvg_scaled_50"): "d098bf3e72c8944000800f4f6d5d6902173c6d6b9a027e68f62af755ed0a11aa",
    ("elegans", "X_pca_hvg_scaled"): "99b69109de0418296a84cdc98a866d835d3c0a359618485d5226f7a442a13b65",
    ("elegans", "X_pca_hvg_scaled_50"): "a3ac3c3717095ca882f45bfec30409a0e440fe074084ef6c23bbf44fcf9bbc35",
}
EXPECTED_CELLS = {"dicty": 48_983, "elegans": 89_701}
PANELS = {
    "dicty": ("cell_type", "cAMP_genes", "milestones"),
    "elegans": ("cell_type", "lineage_markers", "developmental_markers"),
}
HEADLINE_UCE_COVERAGE = {
    "dicty": (12_548, 12_883),
    "elegans": (19_720, 20_222),
}
DIMENSION_CONDITIONS = {
    "PCA": ("X_pca", "X_pca"),
    "scVI": ("X_scVI", "X_scVI"),
    "scGPT": ("X_scGPT_ortholog_tier12", "X_scGPT_ortholog_tier12_pca100_supp"),
    "Geneformer": ("X_geneformer_ortholog_tier12", "X_geneformer_ortholog_tier12_pca100_supp"),
    "UCE-4L": ("X_uce_4L", "X_uce_4L_pca100_supp"),
}


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
    parser.add_argument(
        "--protein-stem",
        default="protein_space_raw_calibrated",
        help="Stem under manuscript/figures/supplementary.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, *, label: str) -> None:
    if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-12):
        raise AssertionError(f"{label}: {left!r} != {right!r}")


def scaled(value: float) -> float:
    return (float(value) + 1.0) / 2.0


def validate_inlined_supplement_tables() -> None:
    """Confirm generated table bodies and reader-facing supplementary order."""
    manuscript = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    assert r"\input{../results/supplements/" not in manuscript

    source_files = (
        "S1b_tier_sweep.tex",
        "S2_ortholog_restricted_uce.tex",
        "S1c_marker_retention.tex",
        "S4_uce_layer_calibration.tex",
        "S6_pca_preprocessing.tex",
        "S5_avgbio_components.tex",
        "S6_pseudotime_stability.tex",
        "S9_coverage_matched_pseudotime.tex",
        "S10_dimension100_avgbio.tex",
        "S11_dimension100_pseudotime.tex",
        "S12_uce_protein_calibration.tex",
    )
    source = "\n".join((SUPP / filename).read_text(encoding="utf-8") for filename in source_files)

    def table_for(text: str, label: str) -> str:
        marker = rf"\label{{{label}}}"
        position = text.index(marker)
        start = text.rfind(r"\begin{table}", 0, position)
        end = text.index(r"\end{table}", position) + len(r"\end{table}")
        return text[start:end]

    def tabular_blocks(table: str) -> list[str]:
        blocks = re.findall(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            table,
            flags=re.DOTALL,
        )
        return [re.sub(r"\s+", " ", block).strip() for block in blocks]

    expected_order = (
        "tab:s1b",
        "tab:s1c",
        "tab:s8",
        "tab:s7",
        "tab:s2",
        "tab:s9",
        "tab:s5",
        "tab:s6",
        "tab:s10",
        "tab:s11",
        "tab:s4",
        "tab:s12",
    )
    for label in expected_order:
        expected = tabular_blocks(table_for(source, label))
        observed = tabular_blocks(table_for(manuscript, label))
        assert observed == expected, f"inlined supplementary table mismatch: {label}"

    main_text, supplement = manuscript.split(
        r"\section*{Supplementary Material}", maxsplit=1
    )
    physical_order = tuple(
        re.findall(r"\\label\{(tab:[^}]+)\}", supplement)
    )
    assert physical_order == expected_order, (
        "supplementary tables are not in first-citation order",
        physical_order,
        expected_order,
    )

    cited_labels = re.findall(
        r"Supplementary Table~\\ref\{(tab:[^}]+)\}",
        main_text,
    )
    first_citation_order = tuple(dict.fromkeys(cited_labels))
    assert first_citation_order == expected_order, (
        "supplementary tables are not first cited sequentially",
        first_citation_order,
        expected_order,
    )
    assert not re.search(r"Supplementary Tables?~S\d", main_text), (
        "hard-coded supplementary table number remains in manuscript text"
    )


def validate_pca() -> None:
    smoke = load_json(ROOT / "results" / "embeddings" / "pca_smoke_determinism.json")
    assert smoke["seed"] == 42
    for organism, record in smoke["runs"].items():
        assert record["identical"] is True
        assert record["pca100_sha256_run_1"] == record["pca100_sha256_run_2"]
        assert record["pca50_sha256_run_1"] == record["pca50_sha256_run_2"]
        work = ROOT / "data" / "pca" / "work" / organism / "full"
        pca100_path = work / f"{organism}_full_hvg_scaled_100_pca.npy"
        pca50_path = work / f"{organism}_full_hvg_scaled_100_pca50.npy"
        assert sha256_file(pca100_path) == record["pca100_sha256_run_2"]
        assert sha256_file(pca50_path) == record["pca50_sha256_run_2"]

    for organism, n_cells in EXPECTED_CELLS.items():
        path = ROOT / "data" / f"{organism}.h5ad"
        atlas = ad.read_h5ad(path, backed="r")
        try:
            for key, width in (("X_pca", 100), ("X_pca_50", 50)):
                array = np.asarray(atlas.obsm[key])
                assert array.shape == (n_cells, width), (organism, key, array.shape)
                assert sha256_array(array) == RAW_HASHES[(organism, key)], (organism, key)
            for key, width in (("X_pca_hvg_scaled", 100), ("X_pca_hvg_scaled_50", 50)):
                array = np.asarray(atlas.obsm[key])
                assert array.shape == (n_cells, width), (organism, key, array.shape)
                assert np.isfinite(array).all(), (organism, key, "non-finite")
                assert np.all(np.var(array, axis=0) > 0), (organism, key, "zero variance")
                assert sha256_array(array) == PROCESSED_HASHES[(organism, key)], (organism, key)
        finally:
            atlas.file.close()

        manifest = load_json(
            ROOT / "results" / "embeddings" / f"pca_{organism}_hvg_scaled_sensitivity_manifest.json"
        )
        recipe = manifest["variants"][0]["recipes"][0]
        assert recipe["recipe"] == "hvg_scaled"
        assert recipe["n_hvg_genes"] == 2000
        assert recipe["embedding_shape"] == [n_cells, 100]
        assert recipe["pca50_shape"] == [n_cells, 50]
        assert recipe["preprocessing"] == [
            "highly_variable_genes",
            "normalize_total_target_sum_1e4",
            "log1p",
            "scale_max_value_10",
        ]


def validate_dimension_matched_100() -> None:
    expected_parameters = {
        "operation": "PCA applied after cell embedding",
        "n_components": 100,
        "center": True,
        "feature_scaling": False,
        "whiten": False,
        "svd_solver": "randomized",
        "seed": 42,
    }
    expected_pairs = {
        pair for pair in DIMENSION_CONDITIONS.values() if pair[0] != pair[1]
    }
    for organism, n_cells in EXPECTED_CELLS.items():
        smoke = load_json(
            ROOT / "results" / "embeddings" / f"dimension_matched_100_{organism}_smoke_manifest.json"
        )
        full = load_json(
            ROOT / "results" / "embeddings" / f"dimension_matched_100_{organism}_manifest.json"
        )
        assert smoke["mode"] == "smoke" and smoke["smoke_cells"] == 512
        assert full["mode"] == "full" and full["smoke_cells"] == 0
        assert smoke["parameters"] == expected_parameters
        assert full["parameters"] == expected_parameters
        assert {(r["source_key"], r["output_key"]) for r in smoke["records"]} == expected_pairs
        assert {(r["source_key"], r["output_key"]) for r in full["records"]} == expected_pairs
        for record in smoke["records"]:
            assert record["output_shape"] == [512, 100]
            assert record["output_sha256"] == record["repeat_output_sha256"]

        atlas = ad.read_h5ad(ROOT / "data" / f"{organism}.h5ad", backed="r")
        try:
            for record in full["records"]:
                source = np.asarray(atlas.obsm[record["source_key"]])
                output = np.asarray(atlas.obsm[record["output_key"]])
                assert list(source.shape) == record["source_shape"]
                assert output.shape == (n_cells, 100)
                assert list(output.shape) == record["output_shape"]
                assert np.isfinite(output).all()
                assert np.all(np.var(output, axis=0) > 0)
                assert sha256_array(source) == record["source_sha256"]
                assert sha256_array(output) == record["output_sha256"]
                retained = float(record["explained_variance_ratio_sum"])
                assert 0.0 < retained <= 1.0001
        finally:
            atlas.file.close()


def validate_eligibility() -> None:
    payload = load_json(SUPP / "uce_gene_eligibility.json")
    json_sets = {
        organism: set(record["eligible_gene_ids"])
        for organism, record in payload["organisms"].items()
    }
    tsv_sets = {"dicty": set(), "elegans": set()}
    with (SUPP / "uce_gene_eligibility.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            tsv_sets[row["organism"]].add(row["gene_identifier"])
    assert json_sets == tsv_sets

    with (SUPP / "marker_sets_longform.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    for row in rows:
        expected = row["gene_identifier"].upper() in json_sets[row["organism"]]
        observed = row["uce_eligible_exact"].lower() == "true"
        assert observed == expected, row


def validate_headline_coverage() -> None:
    """Keep UCE-4L in the manuscript coverage panels and tie it to manifests."""
    for organism, expected in HEADLINE_UCE_COVERAGE.items():
        manifest = load_json(ROOT / "results" / "embeddings" / f"uce_{organism}_manifest.json")
        observed = (
            max(int(run["n_validated_genes"]) for run in manifest["runs"]),
            max(int(run["n_source_genes"]) for run in manifest["runs"]),
        )
        assert observed == expected, (organism, observed, expected)

        covered, total = observed
        expected_percentage = f"{100.0 * covered / total:.1f}%"
        svg = (
            ROOT
            / "manuscript"
            / "figures"
            / "generated"
            / organism
            / "genome_coverage"
            / f"genome_coverage_{organism}.svg"
        ).read_text(encoding="utf-8")
        for required_text in ("UCE-4L", f"{covered:,}", expected_percentage):
            assert required_text in svg, (organism, required_text)


def avgbio_source(organism: str, run: str, panel: str, key: str) -> float:
    path = ROOT / f"results/biological_benchmarks_{organism}/runs/{run}/{panel}.json"
    return float(load_json(path)[key]["avg_bio"])


def pseudo_source(organism: str, run: str, key: str):
    path = ROOT / f"results/pseudotime_{organism}/runs/{run}/pseudotime.json"
    return load_json(path)[key]


def validate_sensitivity_tables(args: argparse.Namespace) -> None:
    uce_conditions = {
        "UCE-4L": "X_uce_4L",
        "UCE-33L": "X_uce_33L",
    }
    with (SUPP / "S4_uce_layer_calibration.tsv").open(encoding="utf-8") as handle:
        uce_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(uce_rows) == 12
    for row in uce_rows:
        source = avgbio_source(
            row["organism"],
            args.avgbio_main_run,
            row["annotation_panel"],
            uce_conditions[row["condition"]],
        )
        close(row["avgBIO"], source, label="S4 avgBIO")

    with (SUPP / "S6_pca_avgbio.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 6
    for row in rows:
        organism = row["organism"]
        panel = row["annotation_panel"]
        raw = avgbio_source(organism, args.avgbio_main_run, panel, "X_pca")
        processed = avgbio_source(
            organism, args.avgbio_pca_run, panel, "X_pca_hvg_scaled"
        )
        close(row["raw_PCA_avgBIO"], raw, label=f"{organism}/{panel}/raw")
        close(row["preprocessed_PCA_avgBIO"], processed, label=f"{organism}/{panel}/processed")
        close(
            row["delta_preprocessed_minus_raw"],
            processed - raw,
            label=f"{organism}/{panel}/delta",
        )

    table_specs = (
        (
            "S5_uce_pseudotime.tsv",
            {
                "UCE-4L": ("X_uce_4L", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
                "UCE-33L": (
                    "X_uce_33L",
                    args.pseudotime_sensitivity_point_run,
                    args.pseudotime_sensitivity_stability_run,
                ),
            },
        ),
        (
            "S6_pca_pseudotime.tsv",
            {
                "Raw-count PCA": ("X_pca", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
                "Preprocessed PCA": (
                    "X_pca_hvg_scaled",
                    args.pseudotime_sensitivity_point_run,
                    args.pseudotime_sensitivity_stability_run,
                ),
            },
        ),
    )
    for filename, conditions in table_specs:
        with (SUPP / filename).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert len(rows) == 8
        assert {row["metric"] for row in rows} == {"spearman", "lins_ccc_consensus"}
        for row in rows:
            key, point_run, stability_run = conditions[row["condition"]]
            point = pseudo_source(row["organism"], point_run, key)
            stability = pseudo_source(row["organism"], stability_run, key)
            metric = row["metric"]
            low, high = stability[f"{metric}_ci"]
            close(row["point_estimate_rescaled"], scaled(point[metric]), label=filename)
            close(row["stability_low_rescaled"], scaled(low), label=filename)
            close(row["stability_high_rescaled"], scaled(high), label=filename)

    with (SUPP / "S9_coverage_matched_pseudotime.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    assert {row["metric"] for row in rows} == {"spearman", "lins_ccc_consensus"}
    coverage_conditions = {
        "Full eligible gene set": ("X_uce_4L", args.pseudotime_main_point_run, args.pseudotime_main_stability_run),
        "Ortholog-available gene set": ("X_uce_4L_orthoT12", args.pseudotime_expanded_point_run, args.pseudotime_expanded_stability_run),
    }
    for row in rows:
        key, point_run, stability_run = coverage_conditions[row["condition"]]
        point = pseudo_source(row["organism"], point_run, key)
        stability = pseudo_source(row["organism"], stability_run, key)
        metric = row["metric"]
        low, high = stability[f"{metric}_ci"]
        close(row["point_estimate_rescaled"], scaled(point[metric]), label="S9 point")
        close(row["stability_low_rescaled"], scaled(low), label="S9 low")
        close(row["stability_high_rescaled"], scaled(high), label="S9 high")

    with (SUPP / "S10_dimension100_avgbio.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 30
    for row in rows:
        native_key, reduced_key = DIMENSION_CONDITIONS[row["method"]]
        assert row["native_key"] == native_key
        assert row["dimension100_key"] == reduced_key
        native = avgbio_source(row["organism"], args.avgbio_main_run, row["annotation_panel"], native_key)
        reduced_run = (
            args.avgbio_main_run if native_key == reduced_key else args.avgbio_dimension_run
        )
        reduced = avgbio_source(row["organism"], reduced_run, row["annotation_panel"], reduced_key)
        close(row["native_avgBIO"], native, label="S10 native")
        close(row["dimension100_avgBIO"], reduced, label="S10 100D")
        close(row["delta_dimension100_minus_native"], reduced - native, label="S10 delta")

    with (SUPP / "S10_dimension100_variance.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 10
    for row in rows:
        native_key, reduced_key = DIMENSION_CONDITIONS[row["method"]]
        assert row["native_key"] == native_key
        if native_key == reduced_key:
            expected_retained = 1.0
        else:
            manifest = load_json(
                ROOT / "results" / "embeddings" / f"dimension_matched_100_{row['organism']}_manifest.json"
            )
            records = [record for record in manifest["records"] if record["source_key"] == native_key]
            assert len(records) == 1
            expected_retained = records[0]["explained_variance_ratio_sum"]
        close(
            row["explained_variance_ratio_retained"],
            expected_retained,
            label="S10 variance",
        )

    with (SUPP / "S11_dimension100_pseudotime.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 20
    assert {row["metric"] for row in rows} == {"spearman", "lins_ccc_consensus"}
    for row in rows:
        native_key, reduced_key = DIMENSION_CONDITIONS[row["method"]]
        native = pseudo_source(row["organism"], args.pseudotime_main_point_run, native_key)
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
        reduced = pseudo_source(row["organism"], reduced_point_run, reduced_key)
        stability = pseudo_source(row["organism"], reduced_stability_run, reduced_key)
        assert row["point_run"] == reduced_point_run
        assert row["stability_run"] == reduced_stability_run
        metric = row["metric"]
        native_value = scaled(native[metric])
        reduced_value = scaled(reduced[metric])
        low, high = stability[f"{metric}_ci"]
        close(row["native_point_rescaled"], native_value, label="S11 native")
        close(row["dimension100_point_rescaled"], reduced_value, label="S11 100D")
        close(row["delta_dimension100_minus_native"], reduced_value - native_value, label="S11 delta")
        close(row["dimension100_stability_low_rescaled"], scaled(low), label="S11 low")
        close(row["dimension100_stability_high_rescaled"], scaled(high), label="S11 high")

    diagnostic_keys = {
        "Nearest human": "mean_nn_cosine",
        "Assigned ortholog T1": "tier1_assigned_cosine",
        "Assigned ortholog T2": "tier2_assigned_cosine",
        "Assigned ortholog T3": "tier3_assigned_cosine",
    }
    with (SUPP / "S5_protein_diagnostics.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    for row in rows:
        organism = row["organism"]
        prefix = diagnostic_keys[row["diagnostic"]]
        manifests = {}
        for adapter_slug in ("whitening", "macrogene"):
            manifest = load_json(
                ROOT
                / "results"
                / "embeddings"
                / "uce"
                / "esm2_adapter"
                / f"{organism}_{adapter_slug}_global"
                / f"{organism}_{adapter_slug}_global_manifest.json"
            )
            assert manifest["method"] == adapter_slug
            manifests[adapter_slug] = manifest
        close(row["cosine_before"], manifests["whitening"]["metrics"][f"{prefix}_before"], label="protein before")
        close(row["cosine_before"], manifests["macrogene"]["metrics"][f"{prefix}_before"], label="protein before")
        close(row["cosine_after_global_whitening"], manifests["whitening"]["metrics"][f"{prefix}_after"], label="protein global after")
        close(row["cosine_after_macrogene_residual"], manifests["macrogene"]["metrics"][f"{prefix}_after"], label="protein macrogene after")
        if row["diagnostic"] == "Nearest human":
            expected_evaluation = int(manifests["whitening"]["n_source_genes"])
        else:
            tier = row["diagnostic"].rsplit("T", 1)[1]
            with Path(manifests["whitening"]["pair_diagnostics"]).open(encoding="utf-8") as handle:
                expected_evaluation = sum(
                    pair["tier"] == tier for pair in csv.DictReader(handle)
                )
        assert int(row["n_evaluation_records"]) == expected_evaluation

    with (SUPP / "S5_protein_calibration_counts.tsv").open(encoding="utf-8") as handle:
        count_rows = list(csv.DictReader(handle, delimiter="\t"))
    observed_counts = {
        (row["component"], row["organism"]): int(row["n_records"])
        for row in count_rows
    }
    assert observed_counts == {
        ("global_fit_human", "all"): 19790,
        ("global_fit_target", "dicty"): 13209,
        ("macrogene_anchor_pairs", "dicty"): 960,
        ("global_fit_target", "elegans"): 19985,
        ("macrogene_anchor_pairs", "elegans"): 1638,
    }

    calibration_keys = {
        "UCE-4L": "X_uce_4L",
        "UCE-4L global": "X_uce_4L_whitening",
        "UCE-33L": "X_uce_33L",
        "UCE-33L global": "X_uce_33L_whitening",
        "UCE-33L macrogene": "X_uce_33L_macrogene",
    }
    with (SUPP / "S12_uce_protein_calibration.tsv").open(encoding="utf-8") as handle:
        calibration_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(calibration_rows) == 30
    for row in calibration_rows:
        expected_key = calibration_keys[row["condition"]]
        assert row["embedding_key"] == expected_key
        expected = avgbio_source(
            row["organism"],
            args.avgbio_main_run,
            row["annotation_panel"],
            expected_key,
        )
        close(row["avgBIO"], expected, label="S12 avgBIO")

    latex_specs = {
        "S4_uce_layer_calibration": (
            ("S4_uce_layer_calibration.tsv", ("avgBIO",)),
            (
                "S5_uce_pseudotime.tsv",
                (
                    "point_estimate_rescaled",
                    "stability_low_rescaled",
                    "stability_high_rescaled",
                ),
            ),
        ),
        "S5_uce_pseudotime_protein": (
            (
                "S5_uce_pseudotime.tsv",
                (
                    "point_estimate_rescaled",
                    "stability_low_rescaled",
                    "stability_high_rescaled",
                ),
            ),
            (
                "S5_protein_diagnostics.tsv",
                (
                    "cosine_before",
                    "cosine_after_global_whitening",
                    "cosine_after_macrogene_residual",
                ),
            ),
        ),
        "S6_pca_preprocessing": (
            (
                "S6_pca_avgbio.tsv",
                ("raw_PCA_avgBIO", "preprocessed_PCA_avgBIO", "delta_preprocessed_minus_raw"),
            ),
            (
                "S6_pca_pseudotime.tsv",
                (
                    "point_estimate_rescaled",
                    "stability_low_rescaled",
                    "stability_high_rescaled",
                ),
            ),
        ),
        "S9_coverage_matched_pseudotime": (("S9_coverage_matched_pseudotime.tsv", ("point_estimate_rescaled", "stability_low_rescaled", "stability_high_rescaled")),),
        "S10_dimension100_avgbio": (
            ("S10_dimension100_avgbio.tsv", ("native_avgBIO", "dimension100_avgBIO", "delta_dimension100_minus_native")),
            ("S10_dimension100_variance.tsv", ("explained_variance_ratio_retained",)),
        ),
        "S11_dimension100_pseudotime": (("S11_dimension100_pseudotime.tsv", ("native_point_rescaled", "dimension100_point_rescaled", "dimension100_stability_low_rescaled", "dimension100_stability_high_rescaled")),),
        "S12_uce_protein_calibration": (
            ("S12_uce_protein_calibration.tsv", ("avgBIO",)),
            (
                "S5_protein_diagnostics.tsv",
                (
                    "cosine_before",
                    "cosine_after_global_whitening",
                    "cosine_after_macrogene_residual",
                ),
            ),
        ),
    }
    for stem, tsv_specs in latex_specs.items():
        tex_path = SUPP / f"{stem}.tex"
        tex = tex_path.read_text(encoding="utf-8")
        bodies = []
        remainder = tex
        while r"\midrule" in remainder:
            _, remainder = remainder.split(r"\midrule", 1)
            body, remainder = remainder.split(r"\bottomrule", 1)
            bodies.append(body)
        observed = Counter(re.findall(r"(?<![\d])-?\d+\.\d+", "\n".join(bodies)))
        expected_values = []
        for filename, numeric_columns in tsv_specs:
            with (SUPP / filename).open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    for column in numeric_columns:
                        value = float(row[column])
                        if (
                            filename == "S10_dimension100_avgbio.tsv"
                            and column == "delta_dimension100_minus_native"
                            and abs(value) < 0.0005
                        ):
                            value = 0.0
                        expected_values.append(f"{value:.3f}")
        assert observed == Counter(expected_values), (stem, observed - Counter(expected_values), Counter(expected_values) - observed)

    provenance = load_json(SUPP / "sensitivity_supplements_manifest.json")["arguments"]
    for argument, identifier in {
        "avgbio_main_run": args.avgbio_main_run,
        "avgbio_pca_run": args.avgbio_pca_run,
        "pseudotime_main_point_run": args.pseudotime_main_point_run,
        "pseudotime_main_stability_run": args.pseudotime_main_stability_run,
        "pseudotime_sensitivity_point_run": args.pseudotime_sensitivity_point_run,
        "pseudotime_sensitivity_stability_run": args.pseudotime_sensitivity_stability_run,
        "avgbio_dimension_run": args.avgbio_dimension_run,
        "pseudotime_expanded_point_run": args.pseudotime_expanded_point_run,
        "pseudotime_expanded_stability_run": args.pseudotime_expanded_stability_run,
    }.items():
        assert provenance[argument] == identifier, (argument, provenance[argument], identifier)


def validate_protein_projection(stem: str) -> None:
    base = ROOT / "manuscript" / "figures" / "supplementary" / stem
    manifest = load_json(base.with_name(f"{stem}_manifest.json"))
    assert manifest["seed"] == 42
    assert manifest["n_proteins_per_species"] == 5000
    assert manifest["n_components"] == 50
    assert manifest["n_neighbors"] == 15
    close(manifest["min_dist"], 0.1, label="protein min_dist")
    assert manifest["same_target_identifiers_before_and_after"] is True
    assert manifest["human_coordinates_reused_between_panels"] is True
    assert set(manifest["shared_axis_limits"]) == {"x", "y"}
    assert set(manifest["software_versions"]) == {
        "python", "numpy", "scikit-learn", "torch", "umap-learn"
    }
    for path, expected_hash in manifest["input_sha256"].items():
        assert sha256_file(Path(path)) == expected_hash, path

    sampled = {"human": set(), "dicty": set(), "elegans": set()}
    with Path(manifest["sampled_genes"]).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sampled[row["organism"]].add(row["gene"])
    sample_counts = {organism: len(genes) for organism, genes in sampled.items()}
    assert set(sample_counts.values()) == {5000}, sample_counts

    coords = np.load(manifest["coordinate_file"])
    assert set(coords.files) == {
        "human",
        "dicty_raw",
        "elegans_raw",
        "dicty_calibrated",
        "elegans_calibrated",
    }
    for key in coords.files:
        assert coords[key].shape == (5000, 2), (key, coords[key].shape)
        assert np.isfinite(coords[key]).all(), key
    assert manifest["coordinate_shape"] == [25_000, 2]
    assert sha256_array(coords["human"]) == manifest["human_coordinates_sha256"]


def main() -> int:
    args = parse_args()
    validate_inlined_supplement_tables()
    validate_pca()
    validate_dimension_matched_100()
    validate_eligibility()
    validate_headline_coverage()
    validate_sensitivity_tables(args)
    validate_protein_projection(args.protein_stem)
    print("publication revision validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
