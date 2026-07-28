"""Step 02: multi-evidence consensus + per-gene tier assignment.

Combines outputs from steps 01b (OrthoDB OG), 01c (OrthoFinder), 01d (OMA REST)
and 01e (curated literature) into one decision per organism gene.

Tier scheme (per the manuscript plan):

    Tier 1 — Gold:   OrthoFinder 1:1  AND  OrthoDB OG 1:1  AND
                     (OMA 1:1 OR curated reference) — all sources agree.
    Tier 2 — Silver: ≥ 2 of {OrthoFinder 1:1, OrthoDB OG 1:1,
                              OMA 1:1, curated} agree on the same human symbol.
    Tier 3 — Bronze: Single-source support, OR many-to-many calls. Recorded
                     in the sidecar TSV; not emitted to the JSON consumed by
                     foundation models.
    Tier 4 — Drop:   No corroborating evidence (never emitted as a row).

Every (organism_gene, human_symbol) pair gets a tier; ``select_best`` picks
the single (symbol, tier) per organism gene used downstream. Ties at the
top tier are broken by:
    1. OrthoFinder 1:1 wins
    2. higher source count wins
    3. curated match wins
    4. lexicographic symbol (deterministic).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from _utils import (
    ensure_human_gene_info,
    ensure_human_ncbi_annotations,
    iter_orthodb_fasta,
    load_human_symbols,
    load_xrefs_subset,
    sha256_file,
    utc_now,
    write_json,
)
from config import (
    EVIDENCE_SOURCES,
    HUMAN_GENE_INFO,
    HUMAN_GENE_INFO_URL,
    HUMAN_NCBI_ANNOTATIONS,
    HUMAN_NCBI_ANNOTATIONS_NOTE,
    HUMAN_ORTHODB_ID,
    ORGANISMS,
    ORTHODB_CACHE_DIR,
    ORTHODB_GENE_XREFS,
    ORTHODB_RELEASE,
    RESULTS_DIR,
)


XREF_CACHE_DIR = ORTHODB_CACHE_DIR / "xrefs"


@dataclass
class CandidateEvidence:
    """Evidence for a single (organism_gene, human_symbol) candidate pair."""

    orthofinder_type: str = ""        # "1:1", "1:many", "many:1", "many:many", or ""
    orthofinder_orthogroup: str = ""
    orthodb_og_category: str = ""
    orthodb_og_id: str = ""
    oma_rel_type: str = ""
    curated_sources: list[str] = field(default_factory=list)

    def has_orthofinder(self) -> bool:
        return bool(self.orthofinder_type)

    def has_orthodb_og(self) -> bool:
        return bool(self.orthodb_og_category)

    def has_oma(self) -> bool:
        return bool(self.oma_rel_type)

    def is_curated(self) -> bool:
        return bool(self.curated_sources)

    def supporting_sources(self) -> list[str]:
        out = []
        if self.has_orthofinder():
            out.append("orthofinder")
        if self.has_orthodb_og():
            out.append("orthodb_og")
        if self.has_oma():
            out.append("oma")
        if self.is_curated():
            out.append("curated")
        return out

    def strict_flags(self) -> dict[str, bool]:
        return {
            "orthofinder": self.orthofinder_type == "1:1",
            "orthodb_og": self.orthodb_og_category == "1:1",
            "oma": self.oma_rel_type == "1:1",
            "curated": self.is_curated(),
        }


def assign_tier(evidence: CandidateEvidence) -> int:
    """Return tier 1-4 for a single candidate."""

    flags = evidence.strict_flags()
    n_strict = sum(flags.values())
    if flags["orthofinder"] and flags["orthodb_og"] and (flags["oma"] or flags["curated"]):
        return 1
    if n_strict >= 2:
        return 2
    if any(flags.values()) or evidence.supporting_sources():
        return 3
    return 4


# ---------------------------------------------------------------------------
# Identifier resolution helpers
# ---------------------------------------------------------------------------


def load_organism_external_ids(
    *,
    organism: str,
    organism_prefix: str,
    fasta_path: Path,
    xref_type: str | None,
    refresh: bool = False,
) -> dict[str, str]:
    """Resolve organism OrthoDB ids → external gene id (DDB_G* or WBGene*)."""

    if xref_type:
        cache_path = XREF_CACHE_DIR / f"{organism}_{xref_type.lower()}_xrefs.tsv"
        pairs = load_xrefs_subset(
            ORTHODB_GENE_XREFS,
            species_prefixes=[organism_prefix],
            xref_types=[xref_type],
            cache_path=cache_path,
            refresh=refresh,
        )
        return {orthodb_id: external_id for (orthodb_id, _t), external_id in pairs.items()}

    out: dict[str, str] = {}
    for orthodb_id, meta, _seq in iter_orthodb_fasta(fasta_path):
        pub_gene_id = meta.get("pub_gene_id") if isinstance(meta, dict) else None
        if pub_gene_id:
            out[orthodb_id] = pub_gene_id
    return out


def _split_semicolon(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [piece.strip() for piece in text.split(";") if piece.strip()]


# ---------------------------------------------------------------------------
# Source loaders → list of (organism_external_id, human_symbol, evidence_kw)
# ---------------------------------------------------------------------------


def _load_orthofinder(
    path: Path,
    *,
    organism_external: dict[str, str],
    human_symbols: dict[str, str],
) -> list[tuple[str, str, dict[str, str]]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[tuple[str, str, dict[str, str]]] = []
    for _, row in df.iterrows():
        organism_ext = organism_external.get(row["organism_gene"])
        human_sym = human_symbols.get(row["human_gene"])
        if not organism_ext or not human_sym:
            continue
        rows.append(
            (
                organism_ext,
                human_sym,
                {
                    "orthofinder_type": row.get("ortholog_type", ""),
                    "orthofinder_orthogroup": row.get("orthogroup_id", ""),
                },
            )
        )
    return rows


def _load_orthodb_og(
    path: Path,
    *,
    organism_external: dict[str, str],
    human_symbols: dict[str, str],
) -> list[tuple[str, str, dict[str, str]]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[tuple[str, str, dict[str, str]]] = []
    for _, row in df.iterrows():
        organism_ext = organism_external.get(row["organism_orthodb_id"])
        if not organism_ext:
            continue
        og_category = row.get("category", "")
        og_id = row.get("og_id", "")
        for human_orthodb in _split_semicolon(row.get("human_orthodb_ids", "")):
            human_sym = human_symbols.get(human_orthodb)
            if not human_sym:
                continue
            rows.append(
                (
                    organism_ext,
                    human_sym,
                    {
                        "orthodb_og_category": og_category,
                        "orthodb_og_id": og_id,
                    },
                )
            )
    return rows


def _load_oma(
    path: Path,
    *,
    organism_external: dict[str, str],
) -> list[tuple[str, str, dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    rows: list[tuple[str, str, dict[str, str]]] = []
    for _, row in df.iterrows():
        organism_ext = organism_external.get(row["organism_orthodb_id"])
        human_sym = (row.get("human_symbol") or "").strip()
        if not organism_ext or not human_sym:
            continue
        rows.append(
            (
                organism_ext,
                human_sym,
                {"oma_rel_type": row.get("oma_rel_type", "")},
            )
        )
    return rows


def _load_curated(
    path: Path,
) -> list[tuple[str, str, dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    rows: list[tuple[str, str, dict[str, str]]] = []
    for _, row in df.iterrows():
        organism_ext = (row.get("organism_gene_id") or "").strip()
        human_sym = (row.get("human_symbol") or "").strip()
        if not organism_ext or not human_sym:
            continue
        rows.append(
            (
                organism_ext,
                human_sym,
                {"curated_source": row.get("source", "curated")},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    *,
    orthofinder_rows,
    orthodb_og_rows,
    oma_rows,
    curated_rows,
) -> dict[str, dict[str, CandidateEvidence]]:
    """Return ``{organism_gene: {human_symbol: CandidateEvidence}}``."""

    matrix: dict[str, dict[str, CandidateEvidence]] = defaultdict(lambda: defaultdict(CandidateEvidence))

    def _ensure(g: str, s: str) -> CandidateEvidence:
        return matrix[g][s]

    for organism_gene, human_symbol, kw in orthofinder_rows:
        ev = _ensure(organism_gene, human_symbol)
        ev.orthofinder_type = kw.get("orthofinder_type", ev.orthofinder_type)
        ev.orthofinder_orthogroup = kw.get("orthofinder_orthogroup", ev.orthofinder_orthogroup)

    for organism_gene, human_symbol, kw in orthodb_og_rows:
        ev = _ensure(organism_gene, human_symbol)
        # If multiple OGs assign the same pair, prefer the strictest cardinality.
        new_cat = kw.get("orthodb_og_category", "")
        if _category_rank(new_cat) > _category_rank(ev.orthodb_og_category):
            ev.orthodb_og_category = new_cat
            ev.orthodb_og_id = kw.get("orthodb_og_id", ev.orthodb_og_id)

    for organism_gene, human_symbol, kw in oma_rows:
        ev = _ensure(organism_gene, human_symbol)
        new_rel = kw.get("oma_rel_type", "")
        if _category_rank(new_rel) > _category_rank(ev.oma_rel_type):
            ev.oma_rel_type = new_rel

    for organism_gene, human_symbol, kw in curated_rows:
        ev = _ensure(organism_gene, human_symbol)
        src = kw.get("curated_source", "curated")
        if src not in ev.curated_sources:
            ev.curated_sources.append(src)

    return matrix


def _category_rank(value: str) -> int:
    """Strictness ranking — higher is more constraining (1:1 wins)."""

    return {"1:1": 4, "1:many": 3, "1:n": 3, "many:1": 2, "n:1": 2, "many:many": 1, "n:m": 1}.get(value, 0)


def select_best(
    organism_gene: str,
    candidate_map: dict[str, CandidateEvidence],
) -> tuple[str, int, CandidateEvidence, dict[str, int]]:
    """Pick best (human_symbol, tier, evidence) for one organism gene."""

    tiered = {sym: assign_tier(ev) for sym, ev in candidate_map.items()}

    def _key(symbol: str) -> tuple[int, int, int, int, str]:
        ev = candidate_map[symbol]
        flags = ev.strict_flags()
        return (
            tiered[symbol],                                # lower tier wins
            0 if flags["orthofinder"] else 1,              # OrthoFinder 1:1 first
            -len(ev.supporting_sources()),                 # more sources first
            0 if flags["curated"] else 1,                  # curated first
            symbol,                                         # deterministic alpha
        )

    best_symbol = min(candidate_map, key=_key)
    return best_symbol, tiered[best_symbol], candidate_map[best_symbol], tiered


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def build_output_table(
    matrix: dict[str, dict[str, CandidateEvidence]],
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for organism_gene, candidate_map in matrix.items():
        best_symbol, tier, ev, tiered = select_best(organism_gene, candidate_map)
        sources = ev.supporting_sources()
        out_rows.append(
            {
                "organism_gene": organism_gene,
                "human_symbol": best_symbol,
                "tier": tier,
                "n_sources": len(sources),
                "sources": ";".join(sources),
                "orthofinder_type": ev.orthofinder_type,
                "orthofinder_orthogroup": ev.orthofinder_orthogroup,
                "orthodb_og_category": ev.orthodb_og_category,
                "orthodb_og_id": ev.orthodb_og_id,
                "oma_rel_type": ev.oma_rel_type,
                "curated": ";".join(ev.curated_sources),
                "all_candidates_json": json.dumps(
                    {sym: t for sym, t in sorted(tiered.items())}, sort_keys=True
                ),
            }
        )
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(["tier", "organism_gene"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consensus tier assignment across orthology evidence sources."
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=sorted(ORGANISMS),
    )
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--human-xrefs",
        type=Path,
        default=HUMAN_NCBI_ANNOTATIONS,
    )
    parser.add_argument(
        "--human-gene-info",
        type=Path,
        default=HUMAN_GENE_INFO,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ORGANISMS[args.organism]
    organism_id = cfg["orthodb_id"]
    organism_xref = cfg.get("source_xref_type")
    organism_fasta = ORTHODB_CACHE_DIR / "api_cache" / "fasta" / f"{organism_id}.fa"

    print(f"=== Consensus tier assignment for {cfg['label']} ===")

    print("Resolving organism OrthoDB → external gene id...")
    organism_external = load_organism_external_ids(
        organism=args.organism,
        organism_prefix=organism_id,
        fasta_path=organism_fasta,
        xref_type=organism_xref,
    )
    print(f"  organism mapping size: {len(organism_external)}")

    print("Resolving human OrthoDB → HGNC symbol...")
    ensure_human_gene_info(args.human_gene_info, HUMAN_GENE_INFO_URL)
    ensure_human_ncbi_annotations(args.human_xrefs, ORTHODB_GENE_XREFS)
    human_symbols = load_human_symbols(args.human_xrefs, args.human_gene_info)
    print(f"  human mapping size: {len(human_symbols)}")

    of_path = args.input_dir / f"01c_orthofinder_{args.organism}_human.tsv"
    og_path = args.input_dir / f"01b_orthodb_og_{args.organism}.tsv"
    oma_path = args.input_dir / f"01d_oma_orthologs_{args.organism}.tsv"
    curated_path = args.input_dir / f"01e_curated_{args.organism}.tsv"

    print("Loading evidence:")
    of_rows = _load_orthofinder(of_path, organism_external=organism_external, human_symbols=human_symbols)
    print(f"  orthofinder: {len(of_rows)} rows  ({of_path.exists()})")
    og_rows = _load_orthodb_og(og_path, organism_external=organism_external, human_symbols=human_symbols)
    print(f"  orthodb_og:  {len(og_rows)} rows  ({og_path.exists()})")
    oma_rows = _load_oma(oma_path, organism_external=organism_external)
    print(f"  oma:         {len(oma_rows)} rows  ({oma_path.exists()})")
    curated_rows = _load_curated(curated_path)
    print(f"  curated:     {len(curated_rows)} rows  ({curated_path.exists()})")

    matrix = aggregate(
        orthofinder_rows=of_rows,
        orthodb_og_rows=og_rows,
        oma_rows=oma_rows,
        curated_rows=curated_rows,
    )
    df = build_output_table(matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.output_dir / f"02_consensus_{args.organism}.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    tier_counts = df["tier"].value_counts().to_dict() if not df.empty else {}
    source_counts = (
        df["n_sources"].value_counts().sort_index().to_dict() if not df.empty else {}
    )

    inputs = {
        "orthofinder_tsv": str(of_path),
        "orthofinder_sha256": sha256_file(of_path) if of_path.exists() else None,
        "orthodb_og_tsv": str(og_path),
        "orthodb_og_sha256": sha256_file(og_path) if og_path.exists() else None,
        "oma_tsv": str(oma_path),
        "oma_sha256": sha256_file(oma_path) if oma_path.exists() else None,
        "curated_tsv": str(curated_path),
        "curated_sha256": sha256_file(curated_path) if curated_path.exists() else None,
        "human_gene_info_path": str(args.human_gene_info),
        "human_gene_info_url": HUMAN_GENE_INFO_URL,
        "human_gene_info_sha256": (
            sha256_file(args.human_gene_info)
            if args.human_gene_info.exists()
            else None
        ),
        "human_xrefs_path": str(args.human_xrefs),
        "human_xrefs_note": HUMAN_NCBI_ANNOTATIONS_NOTE,
        "human_xrefs_sha256": (
            sha256_file(args.human_xrefs) if args.human_xrefs.exists() else None
        ),
    }

    manifest = {
        "step": "02_consensus",
        "orthodb_release": ORTHODB_RELEASE,
        "organism": args.organism,
        "evidence_sources": EVIDENCE_SOURCES,
        "inputs": inputs,
        "n_organism_genes_with_consensus": int(df["organism_gene"].nunique()) if not df.empty else 0,
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "n_sources_distribution": {str(k): int(v) for k, v in source_counts.items()},
        "output_tsv": str(out_tsv),
        "generated_utc": utc_now(),
    }
    write_json(args.output_dir / f"02_consensus_{args.organism}_manifest.json", manifest)

    print("Tier distribution:")
    for k in sorted(tier_counts):
        print(f"  Tier {k}: {tier_counts[k]}")
    print(f"Wrote: {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
