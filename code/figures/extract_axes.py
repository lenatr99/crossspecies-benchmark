#!/usr/bin/env python3
"""Extract a single matplotlib axes layer (default `axes_1`) from an SVG and
write it as a standalone SVG whose canvas is cropped to that axes' *true*
visual bounds (position + size of the axes, glyphs included) — not the
original figure canvas.

Thin CLI over svg_panel_replace.inkscape_crop_layer: keep only `<g id=LAYER>`
(+ referenced <defs>), then let Inkscape crop the page to the drawing's real
bounding box (text glyph extents included, so tick labels / titles are never
clipped).

Usage:
    python extract_axes.py INPUT.svg [OUTPUT.svg] [--layer axes_1]
                                     [--inkscape /path/to/inkscape]

Requires Inkscape on PATH (or pass --inkscape). The first Inkscape run can take
~1 min while it builds its font cache; later runs are sub-second.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svg_panel_replace import inkscape_crop_layer


def extract_axes(svg_path, out_path=None, layer="axes_1", inkscape=None):
    if out_path is None:
        base, ext = os.path.splitext(svg_path)
        out_path = f"{base}_{layer}{ext}"
    out_path = os.path.abspath(out_path)
    open(out_path, "w").write(inkscape_crop_layer(svg_path, layer, inkscape=inkscape))
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--layer", default="axes_1")
    ap.add_argument("--inkscape", default=None)
    a = ap.parse_args()
    extract_axes(a.input, a.output, layer=a.layer, inkscape=a.inkscape)
