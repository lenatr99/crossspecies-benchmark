"""Preset embedding keys and arguments for ``plot_umaps.py``."""

import sys
from pathlib import Path

# Pull key sets from the repo-wide single source of truth so UMAP presets mean
# exactly the same thing as avgBIO, pseudotime, and manuscript figures.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_sets import (  # noqa: E402
    ADAPTER_SUPP,
    LAYER_SUPP,
    MAIN,
    ORTHO_SUPP,
    PCA_PREPROCESSING_SUPP,
    RAW,
    STANDARDIZED_50,
)

VARIANTS = ("tier1", "tier12", "tier123")

REPRESENTATION_SUFFIXES = {
    "embedding": "",
    "pca50": "_50",
}

UCE_PCA50_KEYS = (
    "X_uce_4L_50",
    "X_uce_4L_orthoT1_50",
    "X_uce_4L_orthoT12_50",
    "X_uce_4L_orthoT123_50",
    "X_uce_4L_whitening_50",
    "X_uce_4L_orthoT1_whitening_50",
    "X_uce_4L_orthoT12_whitening_50",
    "X_uce_4L_orthoT123_whitening_50",
    "X_uce_33L_50",
    "X_uce_33L_orthoT1_50",
    "X_uce_33L_orthoT12_50",
    "X_uce_33L_orthoT123_50",
    "X_uce_33L_whitening_50",
    "X_uce_33L_orthoT1_whitening_50",
    "X_uce_33L_orthoT12_whitening_50",
    "X_uce_33L_orthoT123_whitening_50",
    "X_uce_33L_macrogene_50",
)

PCA_PCA50_KEYS = (
    "X_pca_50",
    "X_pca_ortholog_tier1_50",
    "X_pca_ortholog_tier12_50",
    "X_pca_ortholog_tier123_50",
)

MODEL_PRESETS = {
    "main": {
        "keys": MAIN,
        "output_prefix": "main",
    },
    "layer_supp": {
        "keys": LAYER_SUPP,
        "output_prefix": "layer_supp",
    },
    "ortholog_supp": {
        "keys": ORTHO_SUPP,
        "output_prefix": "ortholog_supp",
    },
    "adapter_supp": {
        "keys": ADAPTER_SUPP,
        "output_prefix": "adapter_supp",
    },
    "pca_preprocessing_supp": {
        "keys": PCA_PREPROCESSING_SUPP,
        "output_prefix": "pca_preprocessing_supp",
    },
    "raw": {
        "keys": RAW,
        "output_prefix": "raw",
    },
    "standardized_50": {
        "keys": STANDARDIZED_50,
        "output_prefix": "standardized_50",
    },
    "geneformer": {
        "template": "X_geneformer_ortholog_{variant}{representation_suffix}",
        "variants": VARIANTS,
        "representations": ("pca50",),
    },
    "pca": {
        "keys": PCA_PCA50_KEYS,
        "variants": VARIANTS,
        "representations": ("pca50",),
    },
    "scvi": {
        "template": "X_scVI_ortholog_{variant}{representation_suffix}",
        "variants": VARIANTS,
        "representations": ("pca50",),
    },
    "scvi_batchcorrected": {
        "template": "X_scVI_batchcorrected_ortholog_{variant}{representation_suffix}",
        "variants": VARIANTS,
        "representations": ("pca50",),
        "output_prefix": "scvi_batchcorrected",
    },
    "scgpt": {
        "template": "X_scGPT_ortholog_{variant}{representation_suffix}",
        "variants": VARIANTS,
        "representations": ("pca50",),
    },
    "uce": {
        "keys": UCE_PCA50_KEYS,
        "output_prefix": "uce",
        "output_dir_template": "{output_prefix}/{dataset}/{layer}",
    },
}
