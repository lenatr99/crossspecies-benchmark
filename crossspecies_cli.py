"""Console entry point for the cross-species benchmark.

Installing this package (``pipx install -e .``) puts an ``xspecies`` command on
your PATH so you can drive the whole workflow like any other tool:

    xspecies workflow
    xspecies embed pca --dataset dicty
    xspecies pseudotime --dataset dicty

It is a thin launcher: it locates ``code/run.sh`` in the repo and hands off to
it, so the shell dispatcher stays the single source of truth for routing each
step to its pinned conda env. Nothing about the heavy model/eval envs leaks into
this CLI's own environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Find the repo root that owns ``code/run.sh``.

    With an editable install (``pip install -e .``) this module still lives at
    the repo root, so ``__file__`` resolves it directly. For a non-editable
    install, set ``XSPECIES_HOME`` to the repo checkout.
    """
    here = Path(__file__).resolve().parent
    if (here / "code" / "run.sh").exists():
        return here
    env = os.environ.get("XSPECIES_HOME")
    if env and (Path(env) / "code" / "run.sh").exists():
        return Path(env)
    raise SystemExit(
        "xspecies: cannot locate code/run.sh.\n"
        "  Install editable from the repo (pipx install -e . / pip install -e .),\n"
        "  or set XSPECIES_HOME to the repo checkout."
    )


def main() -> None:
    run_sh = _repo_root() / "code" / "run.sh"
    # Replace this process with the dispatcher so signals / exit codes pass through.
    os.execvp("bash", ["bash", str(run_sh), *sys.argv[1:]])
