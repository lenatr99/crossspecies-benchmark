"""Step 99: manuscript-facing tables for the consensus pipeline.

Read-only; writes to ``results/orthologs/figures/``.

Outputs per organism:

  * ``99_tier_distribution_<organism>.tsv``  -- # genes per tier, total share.
  * ``99_source_overlap_<organism>.tsv``     -- 4-source agreement matrix
                                                (Venn TSV form: how many genes
                                                are confirmed by 1, 2, 3, 4
                                                sources).
  * ``99_coverage_vs_tier_<organism>.tsv``   -- cumulative # genes kept as the
                                                Tier cutoff is relaxed.
  * ``99_legacy_diff_<organism>.tsv``        -- per-gene diff vs the legacy
                                                ``orthologs_03_find_best.py``
                                                output (only emitted when a
                                                legacy CSV is found in
                                                ``--legacy-input-dir``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from _utils import utc_now, write_json
from config import (
    EVIDENCE_SOURCES,
    ORGANISMS,
    RESULTS_DIR,
)


def tier_distribution(consensus_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(consensus_tsv, sep="\t")
    if df.empty:
        return pd.DataFrame(columns=["tier", "n_genes", "share"])
    counts = df["tier"].value_counts().sort_index()
    total = counts.sum()
    rows = [
        {"tier": int(tier), "n_genes": int(n), "share": float(n / total)}
        for tier, n in counts.items()
    ]
    return pd.DataFrame(rows)


def _source_set(row: pd.Series) -> tuple[bool, bool, bool, bool]:
    sources = set(str(row.get("sources", "") or "").split(";"))
    return (
        "orthofinder" in sources,
        "orthodb_og" in sources,
        "oma" in sources,
        "curated" in sources,
    )


def source_overlap(consensus_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(consensus_tsv, sep="\t")
    if df.empty:
        return pd.DataFrame(columns=["pattern", "n_genes", "n_sources"])
    counts: dict[tuple[bool, bool, bool, bool], int] = {}
    for _, row in df.iterrows():
        pattern = _source_set(row)
        counts[pattern] = counts.get(pattern, 0) + 1
    rows: list[dict[str, object]] = []
    for pattern, n in sorted(counts.items(), key=lambda x: (-sum(x[0]), x[0])):
        label = "+".join(name for flag, name in zip(pattern, EVIDENCE_SOURCES) if flag) or "none"
        rows.append({"pattern": label, "n_genes": int(n), "n_sources": int(sum(pattern))})
    return pd.DataFrame(rows)


def coverage_vs_tier(consensus_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(consensus_tsv, sep="\t")
    if df.empty:
        return pd.DataFrame(columns=["min_tier", "cumulative_n_genes"])
    rows = []
    for cutoff in (1, 2, 3, 4):
        n = int((df["tier"].astype(int) <= cutoff).sum())
        rows.append({"min_tier": cutoff, "cumulative_n_genes": n})
    return pd.DataFrame(rows)


def _load_legacy(input_dir: Path, organism: str) -> pd.DataFrame:
    legacy_path = input_dir / f"02_orthologs_{organism}_human_best_orthologs_1to1.csv"
    if not legacy_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(legacy_path)
    df = df.rename(
        columns={
            "dicty_orthodb_id": "organism_orthodb_id",
            "dicty_gene_name": "organism_gene_name",
        }
    )
    return df[["organism_gene_name", "best_human_gene", "selection_method", "confidence_score"]].rename(
        columns={
            "organism_gene_name": "organism_gene",
            "best_human_gene": "human_symbol",
        }
    )


def legacy_diff(
    consensus_tsv: Path,
    legacy_df: pd.DataFrame,
) -> pd.DataFrame:
    new = pd.read_csv(consensus_tsv, sep="\t")
    if new.empty or legacy_df.empty:
        return pd.DataFrame(columns=["organism_gene", "legacy_symbol", "new_symbol", "new_tier", "reason"])
    new_top = (
        new.sort_values(["organism_gene", "tier"]).drop_duplicates("organism_gene")
    )[["organism_gene", "human_symbol", "tier", "sources"]].rename(
        columns={"human_symbol": "new_symbol", "tier": "new_tier"}
    )
    merged = legacy_df.merge(new_top, on="organism_gene", how="outer")
    merged["legacy_symbol"] = merged["human_symbol"]
    merged = merged.drop(columns=[c for c in ("human_symbol",) if c in merged.columns])

    def _reason(row: pd.Series) -> str:
        if pd.isna(row.get("legacy_symbol")):
            return "added_by_consensus"
        if pd.isna(row.get("new_symbol")):
            return "dropped_by_consensus"
        if row.get("legacy_symbol") != row.get("new_symbol"):
            return "flipped"
        return "unchanged"

    merged["reason"] = merged.apply(_reason, axis=1)
    cols = ["organism_gene", "legacy_symbol", "new_symbol", "new_tier", "sources", "reason"]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].sort_values(["reason", "organism_gene"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-facing report tables for the consensus pipeline."
    )
    parser.add_argument(
        "--organisms",
        nargs="+",
        default=sorted(ORGANISMS),
        help="Organisms to report on.",
    )
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--legacy-input-dir",
        type=Path,
        default=None,
        help="Optional directory holding legacy 02_orthologs_*_best_orthologs_1to1.csv outputs for diff.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for organism in args.organisms:
        consensus_tsv = args.input_dir / f"02_consensus_{organism}.tsv"
        if not consensus_tsv.exists():
            print(f"[{organism}] consensus TSV missing at {consensus_tsv}; skipping.")
            continue
        td = tier_distribution(consensus_tsv)
        so = source_overlap(consensus_tsv)
        cv = coverage_vs_tier(consensus_tsv)

        td.to_csv(args.output_dir / f"99_tier_distribution_{organism}.tsv", sep="\t", index=False)
        so.to_csv(args.output_dir / f"99_source_overlap_{organism}.tsv", sep="\t", index=False)
        cv.to_csv(args.output_dir / f"99_coverage_vs_tier_{organism}.tsv", sep="\t", index=False)

        legacy_df = (
            _load_legacy(args.legacy_input_dir, organism)
            if args.legacy_input_dir is not None
            else pd.DataFrame()
        )
        if not legacy_df.empty:
            diff = legacy_diff(consensus_tsv, legacy_df)
            diff.to_csv(
                args.output_dir / f"99_legacy_diff_{organism}.tsv",
                sep="\t",
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            reason_counts = diff["reason"].value_counts().to_dict()
        else:
            reason_counts = {}

        summary[organism] = {
            "tier_distribution": td.to_dict(orient="records"),
            "source_overlap": so.to_dict(orient="records"),
            "coverage_vs_tier": cv.to_dict(orient="records"),
            "legacy_diff_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        }
        print(f"[{organism}] wrote tables to {args.output_dir}")

    write_json(args.output_dir / "99_summary.json", {
        "step": "99_report",
        "organisms": list(summary.keys()),
        "tables": summary,
        "generated_utc": utc_now(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
