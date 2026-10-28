"""Package the recorded marker list as Supplementary Data 1.

Run: python code/figures/package_marker_supplement.py
Uses only recorded local inputs; does not remap genes or rerun analyses.
"""

import csv
import hashlib
import json
import re
import runpy
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/supplements/marker_sets_longform.tsv"
OUT = ROOT / "manuscript/supplementary_data"
PANELS = {
    "dicty": {
        "cell_type": "Cell type",
        "cAMP_genes": "cAMP-pulse-induced genes",
        "milestones": "Milestones",
    },
    "elegans": {
        "cell_type": "Cell/tissue classes",
        "lineage_markers": "Lineage categories",
        "developmental_markers": "Developmental markers",
    },
}
SPECIES = {"dicty": "Dictyostelium discoideum", "elegans": "Caenorhabditis elegans"}
REFERENCE_KEYS = {
    ("dicty", "Cell type"): "Maruo2004; Parikh2010",
    ("dicty", "cAMP-pulse-induced genes"): "Katoh2021; Westbrook2023; Brimson2025",
    ("dicty", "Milestones"): "Katoh2021; Iranfar2003; Rosengarten2015",
    ("elegans", "Cell/tissue classes"): "Sulston1983; Packer2019",
    ("elegans", "Lineage categories"): "Sulston1983; Packer2019",
    ("elegans", "Developmental markers"): "Sulston1983; Packer2019",
}


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tex(value):
    return "".join({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
                    "{": r"\{", "}": r"\}"}.get(c, c) for c in str(value))


def build_guide(rows, references, guide):
    """Typeset a compact guide with the repository's existing LaTeX toolchain."""
    overview, inventory = [], []
    for org, panels in PANELS.items():
        first = True
        for panel in panels.values():
            selected = [r for r in rows if (r["organism"], r["annotation_panel"]) == (org, panel)]
            sets = list(dict.fromkeys(r["marker_set"] for r in selected))
            species = r"\textit{" + tex(SPECIES[org]) + "}" if first else ""
            overview.append(f"{species} & {tex(panel)} & {len(sets)} & {len(selected)} & "
                            f"{len({r['gene_identifier'] for r in selected})}" + r" \\")
            first = False
            for index, marker_set in enumerate(sets):
                members = [r for r in selected if r["marker_set"] == marker_set]
                label = tex(panel) if index == 0 else ""
                inventory.append(f"{tex('D. discoideum' if org == 'dicty' else 'C. elegans')} & "
                                 f"{label} & {tex(marker_set)} & {len(members)}" + r" \\")

    dictionary = [
        ("organism", "Full scientific species name."),
        ("annotation_panel", "Benchmark annotation category."),
        ("marker_set", "Exact marker-set name used in the analysis definitions."),
        ("marker_gene", r"Original marker entry: dictyBase ID for \textit{D. discoideum}; gene symbol for \textit{C. elegans}."),
        ("gene_identifier", "Recorded source identifier. An unresolved symbol is retained when the cached mapping contains no stable ID."),
        ("identifier_namespace", r"\texttt{dictyBase}, \texttt{WormBase}, or \texttt{unresolved\_symbol}."),
        ("identifier_status", r"\texttt{stable\_id\_recorded} indicates a stable-format ID in the export, not validation against a current database release. \texttt{unresolved\_in\_cached\_mapping} marks a fallback symbol."),
        ("panel_context_references", "Literature associated with the annotation panel; contextual references, not individually verified evidence for each gene."),
        ("reference_keys", "Semicolon-separated keys in the accompanying BibTeX file."),
        ("reference_scope", r"\texttt{panel\_context\_not\_gene\_specific\_evidence}."),
        ("ortholog_tier", "Lowest recorded consensus tier (Tier 1, Tier 2 or Tier 3), or unmapped. This precedes model-vocabulary filtering."),
        ("uce_eligible_recorded", "true/false: membership of the recorded identifier in the exported UCE protein-dictionary/chromosome eligibility intersection."),
        ("notes", "Row-specific identifier caveat; blank where no additional caveat is recorded."),
    ]
    field_rows = "\n".join(r"\texttt{" + tex(k) + "} & " + v + r" \\ \addlinespace[5pt]" for k, v in dictionary)
    source = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm,headheight=14pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,booktabs,tabularx,array,enumitem,fancyhdr,microtype}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{7pt}
\renewcommand{\arraystretch}{1.17}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small Supplementary Data 1}
\fancyhead[R]{\small Marker-gene sets}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\hypersetup{pdftitle={Supplementary Data 1. Marker-gene sets used for annotation construction},pdfauthor={}}
\begin{document}
{\small SUPPLEMENTARY DATA 1}\par
\vspace{3mm}
{\LARGE\bfseries Marker-gene sets used for\\annotation construction}\par
\vspace{2mm}
{\large\textit{Dictyostelium discoideum} and \textit{Caenorhabditis elegans}}\par
\vspace{4mm}
\textbf{Data caption.}
Complete curated marker-set memberships for the six benchmark annotation
panels, with recorded gene identifiers, panel-level literature context,
ortholog tiers and UCE input eligibility. Each row represents one gene's
membership in one marker set. Genes may occur in more than one set or panel.
The table contains 1,204 memberships across 30 marker sets.

\section*{Coverage at a glance}
{\small
\begin{tabularx}{\textwidth}{@{}L{31mm}Xrrr@{}}
\toprule
Organism & Annotation panel & Sets & Rows & Unique IDs\\
\midrule
""" + "\n".join(overview) + r"""
\bottomrule
\end{tabularx}}

Unique identifiers are counted within each panel. These counts and the
recorded retention flags reproduce Supplementary Table S2.

\section*{Using the data}
The accompanying \texttt{Supplementary\_Data\_1\_marker\_genes.tsv} is a UTF-8
tab-delimited table. Import it with Tab as the delimiter and preserve
identifier fields as text. The accompanying BibTeX file supplies the full
references linked by \texttt{reference\_keys}. Column definitions follow on
page 2; the complete marker-set inventory is on page 3.

\section*{Interpretation and identifier caveat}
The references provide context for each annotation panel; they are not
individually verified evidence for every gene or set membership. The
definitions are curated inputs to annotation construction, not a claim
that every listed marker was detected in the atlas.

One entry, \textit{C. elegans} \texttt{opt-2} in the intestine set, has no
stable ID in the cached symbol mapping. Its original symbol and recorded
\texttt{unmapped}/\texttt{false} flags are retained and explicitly flagged.
These flags do not establish biological absence of an ortholog or UCE
eligibility after alias resolution.

\clearpage
\section*{Column definitions}
{\small
\begin{tabularx}{\textwidth}{@{}L{55mm}X@{}}
\toprule
Field & Definition\\
\midrule
""" + field_rows + r"""
\bottomrule
\end{tabularx}}

\section*{Input eligibility}
Tier 1 requires one-to-one agreement from OrthoFinder and OrthoDB plus OMA
or curated support. Tier 2 requires at least two sources agreeing on a
one-to-one target. Tier 3 includes single-source or many-to-many support.
The tier field records the lowest assigned tier; cumulative retention at
Tiers 1--2 includes Tier 1 and Tier 2.

Tier membership precedes Geneformer/scGPT vocabulary filtering and is an
upper bound on model retention. UCE eligibility records input availability,
not expression or contribution to an annotation score. ``Unmapped'' denotes
no assignment in the recorded export, not proof that an ortholog does not
exist.

\clearpage
\section*{Marker-set inventory}
Set names retain the exact analysis identifiers. Counts represent listed
memberships; a gene can contribute to more than one set.

{\small
\begin{tabularx}{\textwidth}{@{}L{24mm}L{37mm}Xr@{}}
\toprule
Organism & Annotation panel & Marker set & Genes\\
\midrule
""" + "\n".join(inventory) + r"""
\bottomrule
\end{tabularx}}

\clearpage
\section*{Provenance and reference scope}
The data were exported from the recorded marker-membership table and
checked against both organism-specific marker-definition modules. Per-set
counts match the six recorded annotation manifests. The identifier mapping,
UCE eligibility flags and panel retention totals were checked against the
corresponding cached exports.

These checks establish consistency with the recorded local inputs, not
independent biological validation of marker curation. The export preserves
the recorded identifiers and analysis flags; no alias remapping or
benchmark recomputation was performed. The accompanying README records
source paths, a source SHA-256 checksum and the rebuild command.

Panel-level references are inherited from the manuscript. They should be
read as contextual literature, not as gene-specific evidence annotations.
\nocite{Maruo2004,Parikh2010,Katoh2021,Westbrook2023,Brimson2025,Iranfar2003,Rosengarten2015,Sulston1983,Packer2019}
\renewcommand{\refname}{References}
\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
"""
    with tempfile.TemporaryDirectory(prefix="marker-guide-") as directory:
        work = Path(directory)
        (work / "guide.tex").write_text(source)
        (work / "references.bib").write_bytes(references.read_bytes())
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "guide.tex"],
            cwd=work, capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:] + result.stderr[-1000:])
        log = (work / "guide.log").read_text()
        if "Overfull" in log:
            raise RuntimeError("\n".join(line for line in log.splitlines() if "Overfull" in line))
        guide.write_bytes((work / "guide.pdf").read_bytes())


def main():
    rows = read_tsv(SOURCE)
    assert rows and len(rows) == len({tuple(r.items()) for r in rows})
    maps = json.loads((ROOT / "results/supplements/geneid_maps.json").read_text())
    eligibility = json.loads((ROOT / "results/supplements/uce_gene_eligibility.json").read_text())
    eligible = {org: set(d["eligible_gene_ids"]) for org, d in eligibility["organisms"].items()}
    retention = {
        (r["organism"], r["panel"]): r
        for r in read_tsv(ROOT / "results/supplements/S1c_marker_retention.tsv")
    }
    summary = []
    for org, panels in PANELS.items():
        definitions = runpy.run_path(str(ROOT / f"code/evaluation/src/gene_lists_{org}.py"))
        for attribute, panel in panels.items():
            selected = [r for r in rows if (r["organism"], r["annotation_panel"]) == (org, panel)]
            expected = Counter((s, g) for s, genes in definitions[attribute].items() for g in genes)
            assert Counter((r["marker_set"], r["marker_gene"]) for r in selected) == expected
            manifest = json.loads(
                (ROOT / f"results/evaluation/{org}_{attribute}_discretize_rank_null_manifest.json").read_text()
            )
            assert manifest["programs"] == {s: len(g) for s, g in definitions[attribute].items()}
            unique = {r["gene_identifier"]: r for r in selected}
            counts = [
                len(unique),
                *[sum(r["ortholog_tier"] in [f"Tier {t}" for t in range(1, k + 1)]
                      for r in unique.values()) for k in (1, 2, 3)],
                sum(r["uce_eligible_exact"] == "true" for r in unique.values()),
            ]
            recorded = retention[org, panel]
            assert counts == [int(recorded[k]) for k in (
                "n_marker_genes", "retained_T1", "retained_T12", "retained_T123", "retained_UCE_exact"
            )]
            summary.append(f"| {SPECIES[org]} | {panel} | {len(definitions[attribute])} | {len(selected)} | {len(unique)} |")

    exported = []
    unresolved = []
    for row in rows:
        org, gene, identifier = row["organism"], row["marker_gene"], row["gene_identifier"]
        if org == "elegans":
            assert maps["elegans_sym2wb"].get(gene, gene) == identifier
        resolved = bool(re.fullmatch(r"DDB_G\d+" if org == "dicty" else r"WBGene\d+", identifier))
        assert (row["uce_eligible_exact"] == "true") == (identifier.upper() in eligible[org])
        if not resolved:
            unresolved.append(f"{SPECIES[org]}: {gene} ({row['marker_set']})")
        exported.append({
            "organism": SPECIES[org],
            "annotation_panel": row["annotation_panel"],
            "marker_set": row["marker_set"],
            "marker_gene": gene,
            "gene_identifier": identifier,
            "identifier_namespace": ("dictyBase" if org == "dicty" else "WormBase") if resolved else "unresolved_symbol",
            "identifier_status": "stable_id_recorded" if resolved else "unresolved_in_cached_mapping",
            "panel_context_references": row["supporting_literature"],
            "reference_keys": REFERENCE_KEYS[org, row["annotation_panel"]],
            "reference_scope": "panel_context_not_gene_specific_evidence",
            "ortholog_tier": row["ortholog_tier"],
            "uce_eligible_recorded": row["uce_eligible_exact"],
            "notes": "" if resolved else (
                "No stable ID in the cached symbol mapping. Original symbol and recorded "
                "ortholog/UCE flags retained; these flags do not establish biological absence "
                "of an ortholog or UCE eligibility after alias resolution."
            ),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "Supplementary_Data_1_marker_genes.tsv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(exported)
    assert read_tsv(table) == exported

    bib = (ROOT / "manuscript/references.bib").read_text()
    keys = {k for value in REFERENCE_KEYS.values() for k in value.split("; ")}
    entries = []
    found = set()
    for entry in re.split(r"(?=@\w+\{)", bib):
        match = re.match(r"@\w+\{([^,]+),", entry)
        if match and match[1] in keys:
            entries.append(entry.strip())
            found.add(match[1])
    assert found == keys
    references = OUT / "Supplementary_Data_1_references.bib"
    references.write_text("\n\n".join(entries) + "\n")
    guide = OUT / "Supplementary_Data_1_README.md"
    guide.write_text("""# Supplementary Data 1. Marker-gene sets used for annotation construction

Complete curated marker-set memberships for the six annotation panels in
Dictyostelium discoideum and Caenorhabditis elegans, with recorded identifiers,
panel-level literature context, ortholog tiers and UCE input eligibility.
One row represents one gene's membership in one marker set. A gene may occur
in multiple sets or panels; row counts are not counts of distinct genes.

## Files and use

- `Supplementary_Data_1_marker_genes.tsv`: UTF-8, tab-delimited data; import into
  Excel or another spreadsheet application using Tab as the delimiter and
  text format for identifiers.
- `Supplementary_Data_1_references.bib`: complete manuscript bibliography
  entries corresponding to `reference_keys`.
- `Supplementary_Data_1_guide.pdf`: typeset caption, column guide, marker-set
  inventory and reference bibliography.
- This guide: caption, column definitions, limitations and provenance.

## Column definitions

| Column | Definition |
| --- | --- |
| organism | Full scientific species name. |
| annotation_panel | Benchmark annotation category. |
| marker_set | Exact marker-set name from the analysis definitions. |
| marker_gene | Original marker entry: dictyBase ID for D. discoideum, gene symbol for C. elegans. |
| gene_identifier | Identifier recorded in the source export; an unresolved symbol is retained where no stable ID was available in the cached mapping. |
| identifier_namespace | dictyBase, WormBase, or unresolved_symbol. |
| identifier_status | stable_id_recorded means a stable-format identifier is recorded, not that it has been revalidated against a current database release. unresolved_in_cached_mapping flags the retained fallback symbol. |
| panel_context_references | Literature associated with the annotation panel. These are contextual references, not individually verified evidence for each gene or set membership. |
| reference_keys | Semicolon-separated keys in the accompanying BibTeX file. |
| reference_scope | Explicitly identifies the references as panel-level context. |
| ortholog_tier | Lowest recorded consensus tier (Tier 1, Tier 2, Tier 3), or unmapped. Tier 1 requires one-to-one agreement from OrthoFinder and OrthoDB plus OMA or curated support; Tier 2 requires at least two sources agreeing on a one-to-one target; Tier 3 includes single-source or many-to-many support. |
| uce_eligible_recorded | true/false for whether the recorded identifier belongs to the exported UCE protein-dictionary/chromosome eligibility intersection. |
| notes | Row-specific identifier caveat; blank where no additional caveat is recorded. |

## Interpretation

Ortholog-tier membership precedes Geneformer/scGPT vocabulary filtering and
does not establish that a marker was used by either model. UCE eligibility
does not establish expression or contribution to an annotation score.
The marker definitions are the curated inputs to annotation construction;
this file does not claim that every listed marker was detected in the atlas.
The `unmapped` value means no tier assignment in the recorded export, not
proof that a biological ortholog does not exist.

The unresolved symbol `opt-2` is retained exactly as recorded. Its stable ID
is absent from the cached symbol mapping; its `unmapped` and `false` flags
reflect that recorded identifier and must not be interpreted as a resolved
gene-level exclusion. No alias correction or analysis recomputation was
performed for this publication export.

## Coverage

| Organism | Annotation panel | Marker sets | Membership rows | Unique recorded identifiers |
| --- | --- | ---: | ---: | ---: |
""" + "\n".join(summary) + f"""

Total: {len(exported):,} membership rows across 30 marker sets and six panels.
Unique counts and recorded retention flags reproduce Supplementary Table S2.

## Provenance and validation

Source: `results/supplements/marker_sets_longform.tsv`.
Source SHA-256: `{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}`.

Memberships were compared with `code/evaluation/src/gene_lists_dicty.py` and
`gene_lists_elegans.py`; per-set counts match the six recorded annotation
manifests in `results/evaluation/`. Identifier mappings were checked against
`results/supplements/geneid_maps.json`; UCE flags were checked against
`uce_gene_eligibility.json`. Panel totals and tier/UCE counts match
`S1c_marker_retention.tsv`. These checks verify consistency with the recorded
local inputs, not independent biological validation of the marker curation.

Rebuild with `python code/figures/package_marker_supplement.py`.
""", encoding="utf-8")
    archive = OUT / "Supplementary_Data_1.zip"
    pdf = OUT / "Supplementary_Data_1_guide.pdf"
    build_guide(rows, references, pdf)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in (table, pdf, guide, references):
            bundle.write(path, arcname=path.name)
    print(f"Validated and packaged {len(exported)} memberships; unresolved: {', '.join(unresolved)}")
    print(archive)


if __name__ == "__main__":
    main()
