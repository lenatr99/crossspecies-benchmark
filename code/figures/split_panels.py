#!/usr/bin/env python3
"""Split composed manuscript SVGs into individual panel files.

Each source SVG (dicty_1, elegans_1) contains groups with ids
"a-panel", "b-panel", ... Every such group is exported, cropped tightly to its
own placed bounding box, into a separate SVG *and* PDF via Inkscape.

Why not just `inkscape --export-id=a-panel --export-id-only`?
    Several panels are nested <svg> viewports (matplotlib exports). Inkscape's
    automatic bounding box reads their *inner* 0,0 coordinates instead of the
    placed position, so those panels export blank. Instead we set the root
    <svg> viewBox to the panel's real placed box and export the page, which
    crops predictably in SVG user units for every panel type.

For a clean, small per-panel file we also remove the *other* panel groups
before export (the raster UMAP panels are several MB each).

Normally this is invoked with ``xspecies assemble-figures --split-panels``.
"""

import argparse
import copy
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from svg_panel_replace import find_inkscape

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_ROOT = ROOT / "manuscript" / "figures"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

PANEL_ID_RE = re.compile(r"^[a-z]-panel$")
LABEL_ID_RE = re.compile(r"^[a-z]-label$")


def build_parent_map(root):
    return {child: parent for parent in root.iter() for child in parent}


def panel_bbox(src: Path, panel_el, panel_id: str, inkscape: str):
    """Return (x, y, w, h) of the panel in root user units.

    Nested <svg> panels carry their placement in x/y/width/height attributes.
    Plain <g> panels have no intrinsic box, so we ask Inkscape to measure them.
    """
    children = list(panel_el)
    if len(children) == 1 and children[0].tag == f"{{{SVG_NS}}}svg":
        inner = children[0]
        try:
            return (
                float(inner.get("x", "0")),
                float(inner.get("y", "0")),
                float(inner.get("width")),
                float(inner.get("height")),
            )
        except (TypeError, ValueError):
            pass  # fall through to Inkscape query

    out = subprocess.run(
        [inkscape, str(src),
         f"--query-id={panel_id}",
         "--query-x", "--query-y", "--query-width", "--query-height"],
        capture_output=True, text=True, check=True,
    )
    x, y, w, h = (float(v) for v in out.stdout.split())
    return x, y, w, h


def make_panel_svg(root_template, panel_id, all_panel_ids, bbox) -> ET.ElementTree:
    """Deep-copy the document, keep only `panel_id`, set the page to its bbox."""
    root = copy.deepcopy(root_template)
    parents = build_parent_map(root)

    for el in list(root.iter()):
        pid = el.get("id")
        if el.tag != f"{{{SVG_NS}}}g" or pid is None:
            continue
        drop_other_panel = pid in all_panel_ids and pid != panel_id
        drop_letter_label = LABEL_ID_RE.match(pid) is not None
        if drop_other_panel or drop_letter_label:
            parents[el].remove(el)

    set_viewbox(root, bbox)
    return ET.ElementTree(root)


def set_viewbox(root, bbox) -> None:
    x, y, w, h = bbox
    root.set("viewBox", f"{x} {y} {w} {h}")
    root.set("width", f"{w}")
    root.set("height", f"{h}")


PNG_DPI = 300

# Inkscape mis-measures composite panels that embed matplotlib <svg> viewports
# (flip transforms confuse its bbox query), so `panel_bbox` can return a box
# several times too tall. We therefore render the panel to a transparent raster
# and read the true content extent back from the pixels.
TRIM_RASTER_W = 1200      # px width used only for content detection
TRIM_PAD = 4.0           # padding kept around detected content, in user units


def content_bbox(svg_path: Path, bbox, inkscape: str):
    """Map the opaque region of a rendered panel back to root user units."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "split-panels: Pillow is required for content-aware panel trimming"
        ) from exc

    x, y, w, h = bbox
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png_path = Path(tmp.name)
    try:
        subprocess.run(
            [inkscape, str(svg_path),
             "--export-area-page",
             "--export-type=png",
             f"--export-width={TRIM_RASTER_W}",
             f"--export-filename={png_path}"],
            check=True, capture_output=True, text=True,
        )
        with Image.open(png_path) as im:
            px_w, px_h = im.size
            opaque = im.convert("RGBA").getbbox()
    finally:
        png_path.unlink(missing_ok=True)

    if opaque is None:
        return bbox
    left, top, right, bottom = opaque
    nx = x + (left / px_w) * w - TRIM_PAD
    ny = y + (top / px_h) * h - TRIM_PAD
    nw = (right - left) / px_w * w + 2 * TRIM_PAD
    nh = (bottom - top) / px_h * h + 2 * TRIM_PAD
    return (nx, ny, nw, nh)


def inkscape_export(
    in_svg: Path, out_path: Path, export_type: str, inkscape: str
) -> None:
    cmd = [inkscape, str(in_svg),
           "--export-area-page",
           f"--export-type={export_type}",
           f"--export-filename={out_path}"]
    if export_type == "png":
        cmd += [f"--export-dpi={PNG_DPI}",
                "--export-background=white",
                "--export-background-opacity=1"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def find_panels(root):
    """Return {panel_id: element} for every *-panel group, in document order."""
    panels = {}
    for el in root.iter(f"{{{SVG_NS}}}g"):
        pid = el.get("id")
        if pid and PANEL_ID_RE.match(pid):
            panels.setdefault(pid, el)
    return panels


def process(src: Path, out_dir: Path, *, inkscape: str) -> None:
    tree = ET.parse(src)
    root = tree.getroot()

    panels = find_panels(root)
    if not panels:
        print(f"  no *-panel groups found in {src.name}, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = set(panels)

    print(f"{src.name}: {len(panels)} panels -> {out_dir}")
    for panel_id, panel_el in panels.items():
        letter = panel_id.split("-")[0]
        bbox = panel_bbox(src, panel_el, panel_id, inkscape)
        panel_tree = make_panel_svg(root, panel_id, all_ids, bbox)

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            panel_tree.write(tmp.name, xml_declaration=True, encoding="utf-8")
            tmp_path = Path(tmp.name)

        base = f"{src.stem}_{letter}"
        try:
            tight = content_bbox(tmp_path, bbox, inkscape)
            set_viewbox(panel_tree.getroot(), tight)
            panel_tree.write(tmp_path, xml_declaration=True, encoding="utf-8")

            for ext in ("svg", "pdf", "png"):
                out_path = out_dir / f"{base}.{ext}"
                print(f"  {panel_id} -> {out_path.name}")
                inkscape_export(tmp_path, out_path, ext, inkscape)
        finally:
            tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        type=Path,
        nargs="*",
        default=[
            DEFAULT_FIGURES_ROOT / "dicty_1.svg",
            DEFAULT_FIGURES_ROOT / "elegans_1.svg",
        ],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_FIGURES_ROOT / "panels",
    )
    parser.add_argument("--inkscape")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inkscape = find_inkscape(args.inkscape)
    for src in args.sources:
        src = src.resolve()
        if not src.exists():
            print(f"skip: {src} does not exist")
            continue
        process(src, args.output_root / src.stem.removesuffix("_1"), inkscape=inkscape)


if __name__ == "__main__":
    main()
