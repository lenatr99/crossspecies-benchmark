"""Generate the ortholog supplementary tables (S2, S1b, S1c) from pinned results.

All inputs already exist in the repo; no embedding/benchmark recompute is needed.

  S2  ortholog-restricted UCE : full UCE vs ortho-T12 UCE vs Geneformer vs scGPT, all panels
  S1b tier sweep              : avgBIO per method at ortholog tiers T1/T12/T123
                                (+ source-gene coverage from the pinned inputs)
  S1c marker-set retention    : % of each avgBIO marker set retained by the
                                tier annotations used for the pinned embeddings

Outputs TSV + LaTeX to results/supplements/.
Run:  python code/figures/make_ortholog_supplements.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AVGBIO_RUN = "20260529_124606"  # pinned avgBIO run (point estimates)
OUT = ROOT / "results" / "supplements"

PANELS = {
    "dicty": ["cell_type", "cAMP_genes", "milestones"],
    "elegans": ["cell_type", "lineage_markers", "developmental_markers"],
}
PANEL_LABEL = {
    "cell_type": "Cell type",
    "cAMP_genes": "cAMP-pulse-induced genes",
    "milestones": "Milestones",
    "developmental_markers": "Developmental markers",
    "lineage_markers": "Lineage categories",
}

# C. elegans uses a more specific label for the cell-type panel in reader-facing tables.
PANEL_LABEL_BY_ORGANISM = {
    "dicty": PANEL_LABEL,
    "elegans": PANEL_LABEL | {"cell_type": "Cell/tissue classes"},
}
ORG_TEX = {
    "dicty": r"\species{D.~discoideum}",
    "elegans": r"\species{C.~elegans}",
}
GENE_LIST_ATTR = {  # panel -> attribute name in gene_lists_<org>.py
    "dicty": {"cell_type": "cell_type", "cAMP_genes": "cAMP_genes", "milestones": "milestones"},
    "elegans": {"cell_type": "cell_type", "developmental_markers": "developmental_markers",
                "lineage_markers": "lineage_markers"},
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def avgbio(org: str, panel: str) -> dict:
    p = ROOT / f"results/biological_benchmarks_{org}/runs/{AVGBIO_RUN}/{panel}.json"
    d = json.load(open(p))
    return {k: v["avg_bio"] for k, v in d.items()}


@lru_cache(maxsize=None)
def embedding_tier(org: str) -> dict:
    """Source gene -> tier stored in the atlas used by the pinned embeddings.

    Reading only ``/var`` avoids loading the multi-gigabyte expression matrix.
    These annotations, rather than a subsequently regenerated consensus table,
    define the actual T1/T12/T123 embedding inputs.
    """
    import h5py
    from anndata.io import read_elem

    id_col = {"dicty": "gene_ids", "elegans": "id"}[org]
    with h5py.File(ROOT / f"data/{org}.h5ad", "r") as handle:
        var = read_elem(handle["var"])

    invalid = {"", "nan", "None", "No_ortholog", "<NA>"}
    out = {}
    for _, row in var.iterrows():
        gene_id = str(row[id_col])
        for tier_id in (1, 2, 3):
            symbol = str(row[f"ortholog_tier_{tier_id}"]).strip()
            if symbol not in invalid:
                out[gene_id] = tier_id
                break
    return out


def pinned_coverage_by_tier(org: str) -> dict:
    """Cumulative source-gene counts recorded for the pinned PCA tier inputs."""
    manifest = json.load(open(ROOT / f"results/embeddings/pca_{org}_manifest.json"))
    variants = {row["variant"]: row for row in manifest["variants"]}
    out = {
        1: int(variants["tier1"]["n_selected_ortholog_symbols"]),
        2: int(variants["tier12"]["n_selected_ortholog_symbols"]),
        3: int(variants["tier123"]["n_selected_ortholog_symbols"]),
    }
    tiers = embedding_tier(org)
    observed = {k: sum(tier <= k for tier in tiers.values()) for k in (1, 2, 3)}
    if observed != out:
        raise ValueError(
            f"Stored {org} tier annotations {observed} do not match pinned manifest {out}"
        )
    return out


def union_genes(dct: dict) -> set:
    g = set()
    for v in dct.values():
        g |= set(v)
    return g


@lru_cache(maxsize=None)
def uce_eligible_genes(org: str) -> set[str]:
    path = OUT / "uce_gene_eligibility.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `xspecies code/embeddings/uce/export_gene_eligibility.py` first."
        )
    payload = json.load(open(path))
    return set(payload["organisms"][org]["eligible_gene_ids"])


SUPPORTING_LITERATURE = {
    "dicty": {
        "cell_type": "Maruo et al. (2004); Parikh et al. (2010)",
        "cAMP_genes": "Katoh et al. (2021); Westbrook et al. (2023); Brimson et al. (2025)",
        "milestones": "Katoh et al. (2021); Iranfar et al. (2003); Rosengarten et al. (2015)",
    },
    "elegans": {
        "cell_type": "Sulston et al. (1983); Packer et al. (2019)",
        "developmental_markers": "Sulston et al. (1983); Packer et al. (2019)",
        "lineage_markers": "Sulston et al. (1983); Packer et al. (2019)",
    },
}


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------------------- #
# S2 - ortholog-restricted UCE
# --------------------------------------------------------------------------- #
def build_s2():
    cols = [
        ("X_uce_4L", "UCE-4L (full)"),
        ("X_uce_4L_orthoT12", "UCE-4L (ortho T12)"),
        ("X_geneformer_ortholog_tier12", "Geneformer"),
        ("X_scGPT_ortholog_tier12", "scGPT"),
    ]
    rows = []
    for org, panels in PANELS.items():
        for panel in panels:
            a = avgbio(org, panel)
            rows.append([org, PANEL_LABEL_BY_ORGANISM[org][panel]] + [a.get(k, float("nan")) for k, _ in cols])

    header = ["organism", "panel"] + [lbl for _, lbl in cols]
    write_tsv(OUT / "S2_ortholog_restricted_uce.tsv", header, rows)

    body = []
    previous_org = None
    for r in rows:
        if previous_org is not None and r[0] != previous_org:
            body.append(r"\addlinespace")
        best = max(r[2:])  # bold row max among the four avgBIO columns
        organism = ORG_TEX[r[0]] if r[0] != previous_org else ""
        cells = [organism, r[1]] + [
            (f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v)) for v in r[2:]
        ]
        body.append(" & ".join(cells) + r" \\")
        previous_org = r[0]
    tex = _latex_table(
        label="tab:s2",
        colspec="ll cccc",
        header=r"Organism & Panel & UCE-4L (full) & UCE-4L (Tier 1--2) & Geneformer & scGPT \\",
        body="\n".join(body),
        caption=(
            r"\textbf{Ortholog-restricted UCE isolates representation from coverage.} "
            r"avgBIO for full UCE, UCE restricted to the stored Tier 1--2 source-gene pool used "
            r"to construct the ortholog-mapped inputs before each model's vocabulary filter, "
            r"and the two ortholog-mapped models. On most panels UCE stays close to its full "
            r"value and at or above the ortholog-mapped models under this coverage-aligned "
            r"comparison, consistent with a contribution from the protein-based "
            r"representation; on \textit{D.~discoideum} cell type the restricted embedding drops "
            r"sharply, consistent with a coverage-driven advantage there. Best value per row in bold."
        ),
    )
    (OUT / "S2_ortholog_restricted_uce.tex").write_text(tex)
    return rows


# --------------------------------------------------------------------------- #
# S1b - tier sweep (avgBIO per method per tier + pinned-input coverage)
# --------------------------------------------------------------------------- #
def build_s1b():
    methods = [
        ("PCA", "X_pca_ortholog_tier{t}"),
        ("scVI", "X_scVI_ortholog_tier{t}"),
        ("Geneformer", "X_geneformer_ortholog_tier{t}"),
        ("scGPT", "X_scGPT_ortholog_tier{t}"),
        ("UCE-4L", "X_uce_4L_orthoT{t}"),
    ]
    tiers = ["1", "12", "123"]
    all_rows = []
    for org, panels in PANELS.items():
        cov = pinned_coverage_by_tier(org)
        cov_map = {"1": cov.get(1), "12": cov.get(2), "123": cov.get(3)}
        # mean avgBIO across the 3 panels for each method x tier
        abio = {panel: avgbio(org, panel) for panel in panels}
        rows = []
        for mlabel, keytmpl in methods:
            vals = []
            for t in tiers:
                key = keytmpl.format(t=t)
                per_panel = [abio[p].get(key) for p in panels if key in abio[p]]
                vals.append(sum(per_panel) / len(per_panel) if per_panel else float("nan"))
            rows.append([org, mlabel] + vals)
        all_rows.append((org, cov_map, rows))

    # TSV includes both the input-count row and the avgBIO rows shown in LaTeX.
    header = ["organism", "row", "T1", "T1_2", "T1_3", "value_type"]
    flat = []
    for org, cov_map, rows in all_rows:
        flat.append([org, "Mapped source genes", cov_map["1"], cov_map["12"],
                     cov_map["123"], "count"])
        flat.extend([r + ["avgBIO"] for r in rows])
    write_tsv(OUT / "S1b_tier_sweep.tsv", header, flat)

    # single combined LaTeX table, one organism block after another
    body = []
    for org, cov_map, rows in all_rows:
        body.append(r"\multicolumn{4}{@{}l}{" + ORG_TEX[org] + r"} \\")
        body.append(f"\\quad Mapped source genes (recorded input) & "
                    f"{cov_map['1']} & {cov_map['12']} & {cov_map['123']} \\\\")
        for r in rows:
            body.append(f"\\quad {r[1]} & " + " & ".join(fmt(v) for v in r[2:]) + r" \\")
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    tex = _latex_table(
        label="tab:s1b",
        colspec="l ccc",
        header=r"Method / coverage & Tier 1 & Tiers 1--2 & Tiers 1--3 \\",
        body="\n".join(body),
        caption=(
            r"\textbf{Ortholog-tier sweep.} For each organism, the first row gives the cumulative "
            r"number of atlas source-gene features selected by the stored Tier 1, Tier 1--2 and "
            r"Tier 1--3 annotations used to construct the recorded embedding inputs, before "
            r"model-specific identifier and vocabulary filtering. The remaining rows give avgBIO "
            r"(mean across the organism's three annotation panels) for each method restricted to "
            r"that tier pool. Gene coverage rises several-fold from Tier 1 to Tiers 1--3, whereas "
            r"avgBIO improves only modestly and with diminishing returns, indicating that adding "
            r"lower-confidence orthologs recovers little additional biological signal."
        ),
    )
    (OUT / "S1b_tier_sweep.tex").write_text(tex)
    return all_rows


# --------------------------------------------------------------------------- #
# S1c - marker-set retention by tier
# --------------------------------------------------------------------------- #
def build_s1c():
    # elegans symbol -> WBGene, from cached maps if present else fall back to consensus symbols
    sym2wb = {}
    maps_path = Path("/private/tmp") / "geneid_maps.json"
    # prefer the scratch cache created during setup
    for cand in [maps_path, ROOT / "results" / "supplements" / "geneid_maps.json"]:
        if cand.exists():
            sym2wb = json.load(open(cand)).get("elegans_sym2wb", {})
            break

    gl = {
        "dicty": load_module(ROOT / "code/evaluation/src/gene_lists_dicty.py"),
        "elegans": load_module(ROOT / "code/evaluation/src/gene_lists_elegans.py"),
    }
    rows = []
    long_rows = []
    for org, panels in PANELS.items():
        tier = embedding_tier(org)
        eligible = uce_eligible_genes(org)
        for panel in panels:
            marker_sets = getattr(gl[org], GENE_LIST_ATTR[org][panel])
            genes = union_genes(marker_sets)
            # Map marker symbols to the source-gene IDs stored in the atlas.
            if org == "elegans" and sym2wb:
                ids = {sym2wb.get(g, g) for g in genes}
            else:
                ids = set(genes)
            n = len(ids)
            t1 = sum(1 for g in ids if tier.get(g, 99) <= 1)
            t12 = sum(1 for g in ids if tier.get(g, 99) <= 2)
            t123 = sum(1 for g in ids if tier.get(g, 99) <= 3)
            uce = sum(1 for g in ids if str(g).upper() in eligible)
            rows.append([org, PANEL_LABEL_BY_ORGANISM[org][panel], n, t1, t12, t123, uce])
            for marker_set, marker_genes in marker_sets.items():
                for marker_gene in marker_genes:
                    gene_id = sym2wb.get(marker_gene, marker_gene) if org == "elegans" else marker_gene
                    tier_value = tier.get(gene_id)
                    long_rows.append([
                        org,
                        PANEL_LABEL_BY_ORGANISM[org][panel],
                        marker_set,
                        marker_gene,
                        gene_id,
                        SUPPORTING_LITERATURE[org][panel],
                        f"Tier {tier_value}" if tier_value is not None else "unmapped",
                        str(str(gene_id).upper() in eligible).lower(),
                    ])

    header = [
        "organism", "panel", "n_marker_genes", "retained_T1", "retained_T12",
        "retained_T123", "retained_UCE_exact",
    ]
    write_tsv(OUT / "S1c_marker_retention.tsv", header, rows)
    write_tsv(
        OUT / "marker_sets_longform.tsv",
        [
            "organism", "annotation_panel", "marker_set", "marker_gene",
            "gene_identifier", "supporting_literature", "ortholog_tier",
            "uce_eligible_exact",
        ],
        long_rows,
    )

    body = []
    previous_org = None
    for r in rows:
        if previous_org is not None and r[0] != previous_org:
            body.append(r"\addlinespace")
        n = r[2]
        pct = lambda x: f"{x} ({100*x/n:.0f}\\%)" if n else "0"
        organism = ORG_TEX[r[0]] if r[0] != previous_org else ""
        body.append(f"{organism} & {r[1]} & {n} & {pct(r[3])} & {pct(r[4])} & {pct(r[5])} & "
                    f"{pct(r[6])} \\\\")
        previous_org = r[0]
    tex = _latex_table(
        label="tab:s1c",
        colspec="ll ccccc",
        header=(r"Organism & Panel & Markers & Ortho T1 & Ortho T1--2 & Ortho T1--3 & Protein (UCE) \\"),
        body="\n".join(body),
        caption=(
            r"\textbf{Ortholog mapping discards curated marker genes.} For each avgBIO annotation "
            r"panel, the number of unique marker genes and how many were retained by the stored Tier 1, "
            r"Tier 1--2 and Tier 1--3 annotations used for the recorded embeddings. "
            r"The UCE column gives exact eligibility under the same protein-dictionary and "
            r"chromosome-position filters used for chromosome-aware UCE input construction. "
            r"Retention in the tier pool is an upper bound on "
            r"retained-in-model, since Geneformer/scGPT additionally require the human ortholog to "
            r"be in their vocabulary. Tier 3 may include single-source or many-to-many support."
        ),
        resize=True,
    )
    (OUT / "S1c_marker_retention.tex").write_text(tex)
    return rows


# --------------------------------------------------------------------------- #
def _latex_table(label, colspec, header, body, caption, *, resize=False):
    tabular = (
        f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n"
        "\\end{tabular}\n"
    )
    if resize:
        tabular = "\\resizebox{\\textwidth}{!}{%\n" + tabular + "}\n"
    return (
        "\\begin{table}[ht]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{tabular}\\end{{table}}\n"
    )


def write_tsv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(x) if isinstance(x, float) else x for x in r])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = build_s2()
    s1b = build_s1b()
    s1c = build_s1c()
    print(f"[supplements] wrote {OUT}")
    print("\n=== S2 (ortholog-restricted UCE) ===")
    for r in s2:
        print("  ", r[0], r[1], [fmt(v) for v in r[2:]])
    print("\n=== S1c (marker retention: n, T1, T12, T123, UCE exact) ===")
    for r in s1c:
        print("  ", r[0], r[1], r[2:])


if __name__ == "__main__":
    main()
