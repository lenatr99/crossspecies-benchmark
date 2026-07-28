"""Run the standard marker-label evaluation jobs for both datasets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "evaluation" / "assign_marker_labels.py"
EVAL_DIR = ROOT / "code" / "evaluation"


@dataclass(frozen=True)
class Job:
    dataset: str
    data: Path
    gene_lists: Path
    gene_sets: str
    bundle_mode: str = "xor"
    var_names_key: str = ""
    gene_id_col: str = "gene_ids"
    xor_pick: str = "qvalue"
    enabled_by_default: bool = True


JOBS = (
    Job(
        dataset="dicty",
        data=ROOT / "data" / "dicty.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_dicty.py",
        gene_sets="cell_type",
    ),
    Job(
        dataset="dicty",
        data=ROOT / "data" / "dicty.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_dicty.py",
        gene_sets="cAMP_genes",
    ),
    Job(
        dataset="dicty",
        data=ROOT / "data" / "dicty.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_dicty.py",
        gene_sets="milestones",
        xor_pick="qvalue",
    ),
    Job(
        dataset="dicty",
        data=ROOT / "data" / "dicty.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_dicty.py",
        gene_sets="regulon_clusters",
        bundle_mode="or",
        enabled_by_default=False,
    ),
    Job(
        dataset="dicty",
        data=ROOT / "data" / "dicty.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_dicty.py",
        gene_sets="cell_cycle",
        bundle_mode="or",
        enabled_by_default=False,
    ),
    Job(
        dataset="elegans",
        data=ROOT / "data" / "elegans.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_elegans.py",
        gene_sets="cell_type",
        var_names_key="gene_short_name",
        gene_id_col="",
    ),
    Job(
        dataset="elegans",
        data=ROOT / "data" / "elegans.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_elegans.py",
        gene_sets="developmental_markers",
        var_names_key="gene_short_name",
        gene_id_col="",
    ),
    Job(
        dataset="elegans",
        data=ROOT / "data" / "elegans.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_elegans.py",
        gene_sets="lineage_markers",
        var_names_key="gene_short_name",
        gene_id_col="",
    ),
    Job(
        dataset="elegans",
        data=ROOT / "data" / "elegans.h5ad",
        gene_lists=EVAL_DIR / "src" / "gene_lists_elegans.py",
        gene_sets="cell_type_markers",
        var_names_key="gene_short_name",
        gene_id_col="",
        enabled_by_default=False,
    ),
)


def parse_csv(raw: str) -> set[str]:
    return {piece.strip() for piece in raw.split(",") if piece.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "dicty", "elegans"], default="all")
    parser.add_argument("--gene-sets", default="", help="Comma-separated gene-set names. Defaults to the standard enabled set.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional OR/extra jobs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--write-h5ad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--data",
        str(job.data),
        "--gene-lists",
        str(job.gene_lists),
        "--gene-sets",
        job.gene_sets,
        "--bundle-mode",
        job.bundle_mode,
        "--seed",
        str(args.seed),
        "--xor-pick",
        job.xor_pick,
    ]
    if job.var_names_key:
        cmd.extend(["--var-names-key", job.var_names_key])
    cmd.extend(["--gene-id-col", job.gene_id_col])
    if args.progress:
        cmd.append("--progress")
    if args.write_h5ad:
        cmd.append("--write-h5ad")
    return cmd


def main() -> None:
    args = parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    requested_gene_sets = parse_csv(args.gene_sets)

    jobs = [
        job
        for job in JOBS
        if (args.dataset == "all" or job.dataset == args.dataset)
        and (args.include_optional or job.enabled_by_default)
        and (not requested_gene_sets or job.gene_sets in requested_gene_sets)
    ]
    if not jobs:
        raise SystemExit("No evaluation jobs selected.")

    missing = sorted({str(job.gene_lists) for job in jobs if not job.gene_lists.exists()})
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "Missing gene-list module(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nAdd them later, then rerun this preset script."
        )
    if missing:
        print(
            "WARN: missing gene-list module(s); dry-run will still print commands:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    for job in jobs:
        cmd = command_for(job, args)
        print("\n=== Evaluation:", job.dataset, job.gene_sets, "===")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
