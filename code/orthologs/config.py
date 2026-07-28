"""Shared configuration for the multi-evidence ortholog consensus pipeline.

The pipeline combines four independent ortholog-inference sources:
    1. OrthoDB pair API           (01a)
    2. OrthoDB orthogroup bulk    (01b)
    3. OrthoFinder phylogenetic   (01c)  -- gene-tree based, the gold standard
    4. OMA REST API               (01d)
    5. Curated literature         (01e)  -- OrthoList (Ce), dictyBase (Dicty)

Per-gene candidates are scored by source agreement in 02_consensus, then
binned into Tiers 1-4 (gold/silver/bronze/drop). Only Tier 1-2 land in the
JSON consumed by foundation models; Tier 3 stays in the sidecar.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "orthologs"

ORTHODB_RELEASE = "odb12v1"
ORTHODB_API_VERSION = "v12"
ORTHODB_CACHE_DIR = DATA_DIR / "orthodb" / ORTHODB_RELEASE
ORTHODB_API_BASE = "https://data.orthodb.org"

# Bulk OG-membership flat files live under OrthoDB's rolling ``current`` mirror.
# Filenames use OrthoDB's dump revision tag (``odb12v2_*``), distinct from the
# API/cache snapshot folder name ``odb12v1`` under ``ORTHODB_CACHE_DIR``.
ORTHODB_BULK_BASE = "https://data.orthodb.org/current/download/odb_data_dump"
ORTHODB_BULK_FILE_TAG = "odb12v2"

# Full gene xref table (~4.7 GB gzip) from the same dump as ``odb12v2_*`` bulk files.
# OrthoDB currently ships ``odb12v2_gene_xrefs.tab.gz``; older checkouts may use
# ``odb12v1_gene_xrefs.tab.gz`` as the on-disk name.
ORTHODB_GENE_XREFS_DUMP = DATA_DIR / f"{ORTHODB_BULK_FILE_TAG}_gene_xrefs.tab.gz"
ORTHODB_GENE_XREFS_LEGACY = DATA_DIR / "odb12v1_gene_xrefs.tab.gz"


def resolve_orthodb_gene_xrefs_path() -> Path:
    """Return the gene-xrefs gzip on disk, preferring the current OrthoDB dump filename."""
    if ORTHODB_GENE_XREFS_DUMP.exists():
        return ORTHODB_GENE_XREFS_DUMP
    if ORTHODB_GENE_XREFS_LEGACY.exists():
        return ORTHODB_GENE_XREFS_LEGACY
    return ORTHODB_GENE_XREFS_DUMP


ORTHODB_GENE_XREFS = resolve_orthodb_gene_xrefs_path()

ORTHODB_OG2GENES = ORTHODB_CACHE_DIR / "bulk" / f"{ORTHODB_BULK_FILE_TAG}_OG2genes.tab.gz"
ORTHODB_LEVELS = ORTHODB_CACHE_DIR / "bulk" / f"{ORTHODB_BULK_FILE_TAG}_levels.tab.gz"
ORTHODB_GENES_TAB = ORTHODB_CACHE_DIR / "bulk" / f"{ORTHODB_BULK_FILE_TAG}_genes.tab.gz"

# NCBI Gene symbol authority used to resolve human NCBI GeneID -> approved symbol.
# The checked-in/local file is the uncompressed form of this gzip:
#   curl -L <url> -o Homo_sapiens.gene_info.gz && gunzip Homo_sapiens.gene_info.gz
HUMAN_GENE_INFO = DATA_DIR / "Homo_sapiens.gene_info"
HUMAN_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)

# Existing OrthoDB-human bridge table: human OrthoDB id -> NCBI GeneID.
HUMAN_NCBI_ANNOTATIONS = DATA_DIR / "human_ncbi_annotations.txt"
HUMAN_NCBI_ANNOTATIONS_NOTE = (
    "Derived from OrthoDB gene xrefs by retaining rows for species 9606_1 "
    "with xref type NCBIgid."
)

HUMAN_ORTHODB_ID = "9606_1"
HUMAN_TAXID = "9606"

EUKARYOTA_CLADE = "2759"
METAZOA_CLADE = "33208"
VERTEBRATA_CLADE = "7742"

OMA_API_BASE = "https://omabrowser.org/api"
OMA_CACHE_DIR = DATA_DIR / "oma" / "api_cache"

ORTHOFINDER_BIN = "orthofinder"
ORTHOFINDER_RESULTS_DIR = DATA_DIR / "orthofinder"

CURATED_DIR = DATA_DIR / "curated"
# Curated Ce–human OrthoList (downloaded `.xlsx` from Shaye lab; CSV fallbacks OK).
ORTHOLIST2_PATH = CURATED_DIR / "ortholist_master.xlsx"
# Raw dictyBase ortholog export (tab; see ``orthologs_01e_curated.parse_dictybase``).
DICTYBASE_ORTHOLOG_SOURCE = CURATED_DIR / "dictybase_ortholog_information.txt"
DICTYBASE_ENSP_UNIPROT_CACHE = CURATED_DIR / "dictybase_ensembl_protein_to_uniprot.tsv"
# Faster path: Ensembl REST batch (protein → transcript → gene ``display_name``).
DICTYBASE_ENSP_SYMBOL_CACHE = CURATED_DIR / "dictybase_ensembl_protein_to_symbol.tsv"
# Legacy alias (older pipeline expected a pre-made 2‑column TSV); default is the download path.
DICTYBASE_CURATED_PATH = DICTYBASE_ORTHOLOG_SOURCE

DICTYBASE_ORTHOLOG_INFO_URL = (
    "http://dictybase.org/db/cgi-bin/dictyBase/download/download.pl"
    "?area=general&ID=ortholog_information.txt"
)

HUMAN_PROTEOME_FASTA = ORTHODB_CACHE_DIR / "api_cache" / "fasta" / f"{HUMAN_ORTHODB_ID}.fa"

EVIDENCE_SOURCES = ["orthofinder", "orthodb_og", "oma", "curated"]

ORGANISMS = {
    "dicty": {
        "orthodb_id": "352472_0",
        "taxid": "352472",
        "label": "D. discoideum",
        "default_clade": EUKARYOTA_CLADE,
        "og_level": EUKARYOTA_CLADE,
        "oma_available": True,
        # OrthoDB v12 exposes almost no dicty UniProt xrefs; per-protein
        # ``…/orthologs/`` also omits distant species. We use genome-pair tiles
        # ``pairs/DICDI/9606/`` keyed by FASTA ``pub_gene_id``.
        "oma_queries_from": "pairs_vs_human",
        "oma_pair_genomes": ("DICDI", "9606"),
        "curated_reference": DICTYBASE_CURATED_PATH,
        "mapping_output": DATA_DIR / "ortholog_mapping_orthodb.json",
    },
    "elegans": {
        "orthodb_id": "6239_0",
        "taxid": "6239",
        "label": "C. elegans",
        "default_clade": EUKARYOTA_CLADE,
        "og_level": METAZOA_CLADE,
        "oma_available": True,
        "curated_reference": ORTHOLIST2_PATH,
        "mapping_output": DATA_DIR / "ortholog_mapping_elegans_orthodb.json",
        "source_xref_type": "ENSEMBL",
    },
}
