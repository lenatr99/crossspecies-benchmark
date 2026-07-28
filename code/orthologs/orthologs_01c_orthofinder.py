"""Step 01c: phylogenetic ortholog inference with OrthoFinder.

The core sequence/phylogenetic step of the consensus pipeline. OrthoFinder
infers orthogroups, gene trees, the rooted species tree, and final orthologs
from raw proteomes; per the manuscript we prefer it over reciprocal-best-hit
heuristics because RBH is misled by variable evolution rates between
distantly related species (Emms & Kelly 2019, Fig. 1).

The script:
    1. Stages the human + organism FASTAs (one canonical OrthoDB protein per
       gene) into a clean working directory with deterministic
       ``>{species}|{orthodb_id}`` headers.
    2. Runs ``orthofinder -f <work> -o <out> -S diamond_ultra_sens
       -M msa -A mafft -T fasttree -t <threads>``.
    3. Parses ``Orthologues/Orthologues_<organism>/<organism>__v__<human>.tsv``
       into per-(organism_gene, human_gene) rows tagged with the
       OrthoFinder-derived ortholog cardinality.

OrthoFinder must be installed and on ``$PATH`` (or pointed to by
``--orthofinder-bin``). The script captures full stdout/stderr and embeds
tool versions in the manifest for full reproducibility.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from _utils import fetch_text, sha256_file, sha256_text, utc_now, write_json, write_renamed_fasta
from config import (
    HUMAN_ORTHODB_ID,
    HUMAN_PROTEOME_FASTA,
    ORGANISMS,
    ORTHODB_API_BASE,
    ORTHODB_API_VERSION,
    ORTHODB_CACHE_DIR,
    ORTHOFINDER_BIN,
    ORTHOFINDER_RESULTS_DIR,
    RESULTS_DIR,
)


def _orthodb_fasta_path(species_id: str) -> Path:
    return ORTHODB_CACHE_DIR / "api_cache" / "fasta" / f"{species_id}.fa"


def ensure_orthodb_fasta(species_id: str, cache_path: Path) -> Path:
    """Ensure OrthoDB FASTA cache exists for ``species_id``.

    ``01a`` fetches organism FASTAs as a side effect of counting genes, but the
    human FASTA is not needed there. OrthoFinder needs all proteomes, so this
    step owns the invariant that every staged species has a cached FASTA.
    """

    if cache_path.exists():
        return cache_path

    url = f"{ORTHODB_API_BASE}/{ORTHODB_API_VERSION}/fasta"
    params = {"species": species_id}
    print(f"Missing OrthoDB FASTA cache for {species_id}; downloading {url} ...")
    _status, content = fetch_text(url, params=params, timeout=300, sleep_seconds=0.5)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content, encoding="utf-8")
    write_json(
        cache_path.with_suffix(cache_path.suffix + ".metadata.json"),
        {
            "orthodb_api_version": ORTHODB_API_VERSION,
            "url": url,
            "params": params,
            "retrieved_utc": utc_now(),
            "sha256": sha256_text(content),
            "bytes": cache_path.stat().st_size,
        },
    )
    return cache_path


def prepare_proteomes(
    *,
    organism_species_id: str,
    human_species_id: str,
    proteome_dir: Path,
) -> dict[str, dict[str, object]]:
    """Stage renamed FASTAs for OrthoFinder; return per-species record summary."""

    proteome_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    plan = [
        ("human", human_species_id, HUMAN_PROTEOME_FASTA),
        ("organism", organism_species_id, _orthodb_fasta_path(organism_species_id)),
    ]
    for species_tag, species_id, src in plan:
        ensure_orthodb_fasta(species_id, src)
        dst = proteome_dir / f"{species_tag}.fa"
        n_in, n_out = write_renamed_fasta(src, dst, species_tag=species_tag)
        summary[species_tag] = {
            "species_id": species_id,
            "src": str(src),
            "src_sha256": sha256_file(src),
            "renamed_path": str(dst),
            "renamed_sha256": sha256_file(dst),
            "records_in": n_in,
            "records_out": n_out,
        }
    return summary


def run_orthofinder(
    proteome_dir: Path,
    *,
    out_dir: Path,
    threads: int,
    bin_path: str,
    extra_args: list[str] | None = None,
    log_path: Path,
) -> Path:
    """Invoke OrthoFinder; return path to its ``Results_<date>`` directory."""

    resolved_bin = shutil.which(bin_path)
    if resolved_bin is None:
        raise RuntimeError(
            f"OrthoFinder binary '{bin_path}' not found on PATH. "
            "Install via `conda install -c bioconda orthofinder` or set ORTHOFINDER_BIN in config.py."
        )

    # OrthoFinder rejects an explicitly provided non-default output directory
    # if it already exists. Create only the parent; callers archive incomplete
    # previous runs before reaching this point.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise RuntimeError(f"OrthoFinder output directory is not empty: {out_dir}")
        out_dir.rmdir()
    cmd = [
        sys.executable,
        resolved_bin,
        "-f",
        str(proteome_dir),
        "-o",
        str(out_dir),
        "-n",
        "consensus",
        "-S",
        "diamond_ultra_sens",
        "-M",
        "msa",
        "-A",
        "mafft",
        "-T",
        "fasttree",
        "-t",
        str(threads),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("Running:", " ".join(cmd))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {' '.join(cmd)}\n")
        log.write(f"# started_utc: {utc_now()}\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stdout_parts: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            stdout_parts.append(line)
        returncode = process.wait()
        log.write(f"\n# finished_utc: {utc_now()}\n")
        log.write(f"# exit_code: {returncode}\n")
    stdout_text = "".join(stdout_parts)
    if returncode != 0 or "ERROR:" in stdout_text:
        raise RuntimeError(
            f"OrthoFinder exited with code {returncode}. See {log_path}"
        )

    results = sorted(
        (p for p in out_dir.iterdir() if p.name.startswith("Results_")),
        key=lambda p: p.stat().st_mtime,
    )
    if not results:
        raise RuntimeError(f"No Results_* directory produced under {out_dir}")
    return results[-1]


def is_complete_orthofinder_result(results_dir: Path) -> bool:
    return any((results_dir / "Orthologues").glob("Orthologues_*/*__v__*.tsv"))


def archive_incomplete_run_dir(run_dir: Path) -> None:
    """Move stale/incomplete OrthoFinder run dir aside so OrthoFinder can rerun."""

    if not run_dir.exists():
        return
    if not any(run_dir.iterdir()):
        run_dir.rmdir()
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = run_dir.with_name(f"{run_dir.name}.incomplete_{stamp}")
    print(f"Archiving incomplete OrthoFinder run: {run_dir} -> {archived}")
    shutil.move(str(run_dir), str(archived))


def _detect_orthofinder_version(bin_path: str) -> str:
    try:
        resolved_bin = shutil.which(bin_path) or bin_path
        proc = subprocess.run(
            [sys.executable, resolved_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in (proc.stdout + proc.stderr).splitlines():
            if "OrthoFinder" in line and "version" in line.lower():
                return line.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def parse_orthologues(
    results_dir: Path,
    *,
    organism_tag: str = "organism",
    human_tag: str = "human",
) -> pd.DataFrame:
    """Read OrthoFinder outputs and emit one row per cross-species ortholog pair.

    OrthoFinder's pairwise ``Orthologues/*__v__*.tsv`` files may omit
    single-copy orthogroups from the explicit one-to-one table, while the
    official ``Orthogroups_SingleCopyOrthologues.txt`` file records them.
    We therefore parse both sources: single-copy orthogroups become strict
    ``1:1`` evidence, and the pairwise orthologues TSV contributes the
    broader one-to-many / many-to-one / many-to-many relationships.
    """

    candidates = [
        results_dir / "Orthologues" / f"Orthologues_{organism_tag}" / f"{organism_tag}__v__{human_tag}.tsv",
        results_dir / "Orthologues" / f"Orthologues_{human_tag}" / f"{human_tag}__v__{organism_tag}.tsv",
    ]
    tsv_path = next((p for p in candidates if p.exists()), None)
    if tsv_path is None:
        raise FileNotFoundError(
            "Could not find Orthologues TSV. Looked at: "
            + ", ".join(str(p) for p in candidates)
        )

    raw = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    organism_col = next(c for c in raw.columns if c == organism_tag)
    human_col = next(c for c in raw.columns if c == human_tag)
    og_col = "Orthogroup" if "Orthogroup" in raw.columns else raw.columns[0]

    rows: list[dict[str, str]] = parse_single_copy_orthologues(
        results_dir,
        organism_tag=organism_tag,
        human_tag=human_tag,
    )
    for _, row in raw.iterrows():
        org_field = row[organism_col].strip()
        hum_field = row[human_col].strip()
        if not org_field or not hum_field:
            continue
        organism_genes = _split_gene_field(org_field)
        human_genes = _split_gene_field(hum_field)
        if not organism_genes or not human_genes:
            continue
        n_org = len(organism_genes)
        n_hum = len(human_genes)
        if n_org == 1 and n_hum == 1:
            ortholog_type = "1:1"
        elif n_org == 1:
            ortholog_type = "1:many"
        elif n_hum == 1:
            ortholog_type = "many:1"
        else:
            ortholog_type = "many:many"
        for organism_gene in organism_genes:
            for human_gene in human_genes:
                rows.append(
                    {
                        "organism_gene": _strip_species_prefix(organism_gene),
                        "human_gene": _strip_species_prefix(human_gene),
                        "ortholog_type": ortholog_type,
                        "orthogroup_id": row[og_col],
                        "n_organism_in_pair": str(n_org),
                        "n_human_in_pair": str(n_hum),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates().sort_values(["organism_gene", "human_gene"]).reset_index(drop=True)
    return df


def parse_single_copy_orthologues(
    results_dir: Path,
    *,
    organism_tag: str = "organism",
    human_tag: str = "human",
) -> list[dict[str, str]]:
    """Parse OrthoFinder single-copy orthogroups as strict ``1:1`` rows."""

    single_copy_path = (
        results_dir / "Orthogroups" / "Orthogroups_SingleCopyOrthologues.txt"
    )
    orthogroups_path = results_dir / "Orthogroups" / "Orthogroups.tsv"
    if not single_copy_path.exists() or not orthogroups_path.exists():
        return []

    single_copy_ogs = {
        line.strip()
        for line in single_copy_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not single_copy_ogs:
        return []

    raw = pd.read_csv(orthogroups_path, sep="\t", dtype=str).fillna("")
    required = {"Orthogroup", organism_tag, human_tag}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {orthogroups_path}: {sorted(missing)}"
        )

    rows: list[dict[str, str]] = []
    for _, row in raw[raw["Orthogroup"].isin(single_copy_ogs)].iterrows():
        organism_genes = _split_gene_field(row[organism_tag])
        human_genes = _split_gene_field(row[human_tag])
        if len(organism_genes) != 1 or len(human_genes) != 1:
            continue
        rows.append(
            {
                "organism_gene": _strip_species_prefix(organism_genes[0]),
                "human_gene": _strip_species_prefix(human_genes[0]),
                "ortholog_type": "1:1",
                "orthogroup_id": row["Orthogroup"],
                "n_organism_in_pair": "1",
                "n_human_in_pair": "1",
            }
        )
    return rows


def _split_gene_field(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _strip_species_prefix(token: str) -> str:
    """Convert OrthoFinder tokens back to OrthoDB gene ids.

    OrthoFinder normalizes FASTA ids by replacing ``:`` with ``_`` in outputs,
    so ``human|9606_1_000000`` must become ``9606_1:000000`` before consensus
    can resolve it against OrthoDB xref caches.
    """

    gene_id = token.split("|", 1)[1] if "|" in token else token
    if ":" in gene_id:
        return gene_id
    prefix, sep, suffix = gene_id.rpartition("_")
    if sep and prefix and suffix:
        return f"{prefix}:{suffix}"
    return gene_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OrthoFinder on the human + organism proteomes."
    )
    parser.add_argument(
        "--organism",
        default="dicty",
        choices=sorted(ORGANISMS),
        help="Organism to process.",
    )
    parser.add_argument("--human-id", default=HUMAN_ORTHODB_ID)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--orthofinder-bin", default=ORTHOFINDER_BIN)
    parser.add_argument(
        "--proteome-dir",
        type=Path,
        default=None,
        help="Working dir for renamed proteomes. Defaults to <orthofinder>/<organism>/proteomes.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="OrthoFinder run output dir. Defaults to <orthofinder>/<organism>/run.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Skip the OrthoFinder run if a Results_* directory already exists.",
    )
    parser.add_argument(
        "--extra-orthofinder-args",
        nargs="*",
        default=None,
        help="Extra args to forward to OrthoFinder (e.g. --extra-orthofinder-args -y).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ORGANISMS[args.organism]
    organism_species_id = cfg["orthodb_id"]
    work_root = ORTHOFINDER_RESULTS_DIR / args.organism
    proteome_dir = args.proteome_dir or work_root / "proteomes"
    run_dir = args.results_dir or work_root / "run"

    print(f"=== OrthoFinder run for {cfg['label']} vs human ===")
    print(f"Threads: {args.threads}")
    print(f"Proteome dir: {proteome_dir}")
    print(f"Run dir: {run_dir}")

    proteome_summary = prepare_proteomes(
        organism_species_id=organism_species_id,
        human_species_id=args.human_id,
        proteome_dir=proteome_dir,
    )

    log_path = work_root / "orthofinder.log"
    existing_results = sorted(
        (p for p in run_dir.iterdir() if p.name.startswith("Results_")),
        key=lambda p: p.stat().st_mtime,
    ) if run_dir.exists() else []
    complete_results = [p for p in existing_results if is_complete_orthofinder_result(p)]

    if complete_results and args.reuse_results:
        results_dir = complete_results[-1]
        print(f"Reusing existing OrthoFinder results at {results_dir}")
    else:
        if existing_results and args.reuse_results:
            print(
                "Existing OrthoFinder Results_* directories are incomplete "
                "(missing Orthologues TSVs); rerunning."
            )
            archive_incomplete_run_dir(run_dir)
        results_dir = run_orthofinder(
            proteome_dir,
            out_dir=run_dir,
            threads=args.threads,
            bin_path=args.orthofinder_bin,
            extra_args=args.extra_orthofinder_args,
            log_path=log_path,
        )
    print(f"OrthoFinder results dir: {results_dir}")

    df = parse_orthologues(results_dir)
    print(f"Parsed {len(df)} cross-species ortholog rows")
    if not df.empty:
        ortholog_type_counts = df["ortholog_type"].value_counts().to_dict()
        n_one_to_one = int((df["ortholog_type"] == "1:1").sum())
    else:
        ortholog_type_counts = {}
        n_one_to_one = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = args.output_dir / f"01c_orthofinder_{args.organism}_human.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)

    species_tree_path = results_dir / "Species_Tree" / "SpeciesTree_rooted.txt"
    species_tree_newick = species_tree_path.read_text(encoding="utf-8").strip() if species_tree_path.exists() else ""

    manifest = {
        "step": "01c_orthofinder",
        "organism": args.organism,
        "organism_orthodb_id": organism_species_id,
        "human_orthodb_id": args.human_id,
        "orthofinder_bin": args.orthofinder_bin,
        "orthofinder_version": _detect_orthofinder_version(args.orthofinder_bin),
        "threads": args.threads,
        "proteome_dir": str(proteome_dir),
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "log_path": str(log_path),
        "proteomes": proteome_summary,
        "species_tree_newick": species_tree_newick,
        "ortholog_type_counts": ortholog_type_counts,
        "n_one_to_one_pairs": n_one_to_one,
        "n_total_pairs": int(len(df)),
        "output_tsv": str(tsv_out),
        "extra_orthofinder_args": args.extra_orthofinder_args,
        "generated_utc": utc_now(),
    }
    write_json(args.output_dir / f"01c_orthofinder_{args.organism}_manifest.json", manifest)

    print(json.dumps(ortholog_type_counts, indent=2))
    print(f"Wrote: {tsv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
