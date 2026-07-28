"""Choose which timestamped results run is the active ("current") one.

Benchmarks write each run under ``results/<area>_<organism>/runs/<timestamp>/``.
Downstream consumers (the figures) read the run named by a small pointer file,
``runs/CURRENT`` — so nothing silently jumps to a different run when you compute
a new one. Use ``xspecies select`` to set the pointer (``--run latest`` resolves
to the newest at that moment) and ``xspecies runs`` to see what's available.

Pure stdlib so it runs in any environment without a conda env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dicty", "elegans")
# Command kind -> results-area prefix.
KINDS = {
    "avgbio": "biological_benchmarks",
    "pseudotime": "pseudotime",
}
POINTER = "CURRENT"


def area_dir(organism: str, kind: str) -> Path:
    return ROOT / "results" / f"{KINDS[kind]}_{organism}"


def list_runs(area: Path) -> list[str]:
    """Timestamp run names under ``area/runs``, oldest first (lexical = chrono)."""
    runs = area / "runs"
    if not runs.is_dir():
        return []
    return sorted(d.name for d in runs.iterdir() if d.is_dir() and d.name != "__pycache__")


def resolve_run(area: Path, run: str) -> str | None:
    """Map ``latest`` or an explicit timestamp to an existing run name."""
    runs = list_runs(area)
    if not runs:
        return None
    if run == "latest":
        return runs[-1]
    return run if run in runs else None


def get_current(area: Path) -> str | None:
    """The pinned run name, if the pointer exists and still resolves to a dir."""
    pointer = area / "runs" / POINTER
    if pointer.is_file():
        name = pointer.read_text().strip()
        if name and (area / "runs" / name).is_dir():
            return name
    return None


def set_current(area: Path, run: str) -> str:
    name = resolve_run(area, run)
    if name is None:
        avail = ", ".join(list_runs(area)) or "(none)"
        raise SystemExit(f"select: no run '{run}' under {area / 'runs'} (available: {avail})")
    (area / "runs").mkdir(parents=True, exist_ok=True)
    (area / "runs" / POINTER).write_text(name + "\n")
    return name


def current_file(area: Path, filename: str) -> Path | None:
    """Resolve ``filename`` through the pinned run.

    Pinned run -> legacy flat ``area/filename`` -> ``None``. Deliberately does
    *not* auto-pick the latest run; pin one with ``xspecies select``.
    """
    name = get_current(area)
    if name:
        f = area / "runs" / name / filename
        if f.exists():
            return f
    flat = area / filename
    return flat if flat.exists() else None


# ---- CLI ----------------------------------------------------------------
def _datasets(arg: str) -> list[str]:
    return list(DATASETS) if arg == "all" else [arg]


def _kinds(arg: str) -> list[str]:
    return list(KINDS) if arg == "all" else [arg]


def _cmd_set(args: argparse.Namespace) -> None:
    for organism in _datasets(args.dataset):
        for kind in _kinds(args.kind):
            area = area_dir(organism, kind)
            if not (area / "runs").is_dir():
                print(f"[{organism}/{kind}] no runs/ dir at {area} — skipping")
                continue
            name = set_current(area, args.run)
            print(f"[{organism}/{kind}] current -> {name}")


def _cmd_list(args: argparse.Namespace) -> None:
    for organism in _datasets(args.dataset):
        for kind in _kinds(args.kind):
            area = area_dir(organism, kind)
            runs = list_runs(area)
            current = get_current(area)
            print(f"\n{organism} / {kind}  ({area})")
            if not runs:
                print("  (no runs)")
                continue
            for name in runs:
                mark = "* " if name == current else "  "
                print(f"  {mark}{name}")
            if current is None:
                print("  (none pinned — using legacy flat files if present)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("set", help="Pin the current run.")
    s.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    s.add_argument("--kind", choices=[*KINDS, "all"], default="all")
    s.add_argument("--run", default="latest", help="Run timestamp, or 'latest'.")
    s.set_defaults(func=_cmd_set)

    l = sub.add_parser("list", help="List runs and show which is pinned.")
    l.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    l.add_argument("--kind", choices=[*KINDS, "all"], default="all")
    l.set_defaults(func=_cmd_list)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
