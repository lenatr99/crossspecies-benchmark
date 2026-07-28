"""avgBIO embedding-key presets.

Thin re-export of the repo-wide canonical sets in ``code/embedding_sets.py`` so
avgBIO, pseudotime, and the figures all agree on what each set means. Edit the
sets there, not here. See ``code/EMBEDDINGS.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# code/ holds the canonical embedding_sets module (this file is code/evaluation/util/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embedding_sets import (  # noqa: E402
    ADAPTER_SUPP as ADAPTER_SUPP_AVGBIO_EMBEDDINGS,
    DIMENSION_MATCHED_100_SUPP as DIMENSION_MATCHED_100_SUPP_AVGBIO_EMBEDDINGS,
    EMBEDDING_SETS as AVGBIO_EMBEDDING_PRESETS,
    LAYER_SUPP as LAYER_SUPP_AVGBIO_EMBEDDINGS,
    MAIN as MAIN_AVGBIO_EMBEDDINGS,
    ORTHO_SUPP as ORTHO_SUPP_AVGBIO_EMBEDDINGS,
    RAW as RAW_AVGBIO_EMBEDDINGS,
    STANDARDIZED_50 as STANDARDIZED_50_AVGBIO_EMBEDDINGS,
)

UMAP_DERIVED_AVGBIO_EMBEDDINGS = RAW_AVGBIO_EMBEDDINGS
DEFAULT_AVGBIO_EMBEDDINGS = MAIN_AVGBIO_EMBEDDINGS

__all__ = [
    "AVGBIO_EMBEDDING_PRESETS",
    "DEFAULT_AVGBIO_EMBEDDINGS",
    "MAIN_AVGBIO_EMBEDDINGS",
    "LAYER_SUPP_AVGBIO_EMBEDDINGS",
    "ORTHO_SUPP_AVGBIO_EMBEDDINGS",
    "ADAPTER_SUPP_AVGBIO_EMBEDDINGS",
    "DIMENSION_MATCHED_100_SUPP_AVGBIO_EMBEDDINGS",
    "RAW_AVGBIO_EMBEDDINGS",
    "STANDARDIZED_50_AVGBIO_EMBEDDINGS",
    "UMAP_DERIVED_AVGBIO_EMBEDDINGS",
]
