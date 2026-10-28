# Manuscript figure pipeline

Figure generation has two explicit stages:

```bash
# 1. Render data-driven SVG panels from the H5AD and pinned benchmark runs.
xspecies figures --dataset dicty
xspecies figures --dataset dicty --only pseudotime,time

# 2. Fill the Illustrator master and export final SVG/PDF/PNG files.
xspecies assemble-figures --dataset dicty
```

`xspecies` selects the foundation conda environment automatically. Assembly
also requires Inkscape on `PATH` (or `--inkscape /path/to/inkscape`). Graphik
Trial and Graphik Compact Trial are the publication fonts; the command warns
when fontconfig can prove they are unavailable.

## Files

- `make_figures.py` — CLI driver. Opens the H5AD **backed** (the multi-GB
  expression matrix stays on disk; only `obs` and the UMAP `obsm` arrays load),
  derives numeric time from the time labels, and dispatches each panel family.
- `panels.py` — matplotlib primitives: the outlined UMAP scatter (continuous &
  categorical), the 1D pseudotime line plot, the two grouped bar charts, and the
  standalone legend.
- `figure_presets.py` — per-organism config: data path, time column, the
  categorical "gene set" panels and their colours, and the bar-chart inputs.
  Add an organism by adding an entry to `DATASETS`.
- `assemble_figures.py` — repository-root-aware assembly and export driver.
- `svg_panel_replace.py` — reusable Illustrator/Matplotlib SVG replacement
  primitives.
- `split_panels.py` — optional tightly cropped panel SVG/PDF/PNG exports.
- `extract_axes.py` — standalone utility for cropping a Matplotlib axes layer.

## Stage 1 outputs

| Family | Path | Source |
|--------|------|--------|
| `pseudotime` | `pseudotime/<method>.svg` | `obs["<method>_dpt_pseudotime"]` |
| `time` | `time/<method>.svg` | numeric time from `obs["<time_key>"]` |
| `<gene_set>` | `<gene_set>/<method>.svg` (+ `legend.svg`) | categorical `obs` column |
| `line_plot` | `line_plot/1dplot_pseudotime_<method>.svg` | mean time per pseudotime bin |
| `bioconservation` | `bioconservation/bioconservation_metrics.svg` | `results/biological_benchmarks_<org>/<gene_set>.json` |
| `pseudotime_metrics` | `pseudotime/pseudotime_metrics.svg` | manuscript point/stability runs pinned in `figure_presets.py` |

The two bar-chart families are skipped with a message if their JSON inputs are
missing, so the UMAP panels still render. The bioconservation chart reads the
`avgbio` run pinned with `xspecies select`. The pseudotime chart deliberately
combines the full-data point run and repeated-subsampling stability run recorded
in `PSEUDOTIME_RUNS` in `figure_presets.py`.

```bash
xspecies pseudotime --dataset dicty          # writes results/pseudotime_dicty/runs/<ts>/
# Record the reviewed point/stability run IDs in PSEUDOTIME_RUNS before rendering.
```

## Stage 2 inputs and outputs

Tracked layout masters live in:

```text
manuscript/figures/templates/dicty_1.svg
manuscript/figures/templates/elegans_1.svg
```

Assembly reads those masters plus `manuscript/figures/generated/<organism>/` and writes:

```text
manuscript/figures/<organism>_1.svg
manuscript/figures/<organism>_1.pdf
manuscript/figures/<organism>_1.png
```

The manuscript references the PDFs in this directory. Use
`--split-panels` to additionally create ignored, reproducible exports under
`manuscript/figures/panels/<organism>/`.

Useful checks and overrides:

```bash
# Check every required panel without running Inkscape.
xspecies assemble-figures --dataset all --validate-only

# Use non-default locations or an explicit Inkscape installation.
xspecies assemble-figures --dataset dicty \
  --plots-root manuscript/figures/generated \
  --sources-root manuscript/figures/templates \
  --output-root manuscript/figures \
  --inkscape /Applications/Inkscape.app/Contents/MacOS/inkscape
```

The command validates all inputs before writing outputs. Missing aggregate
panels usually mean stage 1 skipped an unpinned benchmark result; run the
benchmark, select its run with `xspecies select`, then rerun both stages.
