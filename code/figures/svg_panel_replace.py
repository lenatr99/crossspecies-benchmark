#!/usr/bin/env python3
"""Reusable core for replacing Illustrator-placed raster panels in an SVG with
external matplotlib SVGs, each dropped at the EXACT same on-canvas bounding box
the raster occupied (same position + size), optionally flipped.

For every target <image id=...> we:
  1. read its width/height + transform="translate() rotate() scale()"
  2. map the 4 corners through that affine -> axis-aligned bounding box
  3. inline the new SVG as a nested <svg> viewport into that box,
     preserveAspectRatio="none" so it fills exactly, plus an optional flip
     about the box centre.

The source is inlined as a *real* nested <svg> (vector) rather than a base64
<image xlink:href="data:image/svg+xml;...">: Inkscape/Illustrator render any
<image> element as a flat bitmap, which made the swapped-in panels look
rasterized. Nested <svg> keeps axes, ticks and text as crisp editable vector.
(Any point cloud that matplotlib itself rasterized into an inner data:image/png
stays a bitmap — that's a source-generation setting, not something we control
here.)

This module is figure-agnostic: callers supply job groups (id -> (file, flip))
and a source directory. See ``assemble_figures.py`` for the repository config.
"""
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

# --- reusable flip transforms -------------------------------------------
# each is one of: None, "h", "v", "hv" (mirror about the box centre),
# or "r90"/"r270" (quarter-turn rotations, refit to the box).
# Reuse these constants across figures/groups so orientation stays consistent.
FLIP_NONE      = None
FLIP_PCA_UMAP  = "v"     # pca & umap panels
FLIP_MODELS    = "hv"    # scGPT / geneformer / uce model panels (reused)
FLIP_ROT180    = "hv"     # scvi-style 180 deg rotation (reused)
FLIP_ROT90     = "r90"    # 90 deg clockwise, rescaled to fit the box
FLIP_ROT270    = "r270"   # 90 deg counter-clockwise, rescaled to fit the box
FLIP_H_ROT90   = "hr90"   # horizontal mirror + 90 deg (mirror applied last)
FLIP_H_ROT270  = "hr270"  # horizontal mirror + 90 deg counter-clockwise


def bbox_of(line):
    """Axis-aligned on-canvas bounding box of a transformed <image>."""
    W = float(re.search(r'width="([\d.]+)"',  line).group(1))
    H = float(re.search(r'height="([\d.]+)"', line).group(1))
    tx, ty = map(float, re.search(r'translate\(([-\d.]+)\s+([-\d.]+)\)', line).groups())
    rot = re.search(r'rotate\(([-\d.]+)\)', line)
    rot = float(rot.group(1)) if rot else 0.0
    scl = re.search(r'scale\(([-\d.]*)\s*([-\d.]*)?\)', line)
    sx = float(scl.group(1)); sy = float(scl.group(2)) if scl.group(2) else sx
    t = math.radians(rot); c, s = math.cos(t), math.sin(t)
    def xf(x, y):
        x, y = sx * x, sy * y                 # scale
        x, y = c * x - s * y, s * x + c * y   # rotate
        return tx + x, ty + y                 # translate
    pts = [xf(0, 0), xf(W, 0), xf(0, H), xf(W, H)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx, by = min(xs), min(ys)
    return bx, by, max(xs) - bx, max(ys) - by


def flip_transform(mode, bx, by, bw, bh):
    """Mirror about the box centre. Returns a transform string ('' if none)."""
    cx, cy = bx + bw / 2, by + bh / 2
    if mode == "h":  return f'translate({2 * cx:.4f} 0) scale(-1 1)'
    if mode == "v":  return f'translate(0 {2 * cy:.4f}) scale(1 -1)'
    if mode == "hv": return f'translate({2 * cx:.4f} {2 * cy:.4f}) scale(-1 -1)'
    # quarter turns about the box centre, optionally combined with a mirror.
    # rotating swaps the panel's width and height, so rescale by (bh/bw, bw/bh)
    # to refit it back into the same box; the mirror doesn't change size.
    # accepted: "r90"/"r270" and mirror-prefixed "hr90"/"vr90"/"hvr90" (and 270).
    m = re.fullmatch(r'(h?v?)r(90|270)', mode or "")
    if m:
        mir, deg = m.group(1), m.group(2)
        sx, sy = bh / bw, bw / bh
        mx = -1 if "h" in mir else 1
        my = -1 if "v" in mir else 1
        rot = "90" if deg == "90" else "-90"
        pre = f'scale({mx} {my}) ' if (mx, my) != (1, 1) else ""
        return (f'translate({cx:.4f} {cy:.4f}) {pre}rotate({rot}) '
                f'scale({sx:.4f} {sy:.4f}) translate({-cx:.4f} {-cy:.4f})')
    return ""


_ID_RE = re.compile(r'\bid="([^"]+)"')


def namespace_ids(svg_text, prefix):
    """Prefix every id defined in `svg_text` (and every #ref to it) with
    `prefix`, so the svg can be inlined into a larger document without id
    collisions across panels (matplotlib reuses ids like `axes_1`, glyph ids
    like `DejaVuSans-63`, and clip ids that can clash between files).
    """
    for i in set(_ID_RE.findall(svg_text)):
        new = prefix + i
        svg_text = svg_text.replace(f'id="{i}"', f'id="{new}"')
        svg_text = svg_text.replace(f'url(#{i})', f'url(#{new})')
        svg_text = svg_text.replace(f'xlink:href="#{i}"', f'xlink:href="#{new}"')
        svg_text = svg_text.replace(f'href="#{i}"', f'href="#{new}"')
    return svg_text


def inline_svg(svg_text, elem_id, bx, by, bw, bh, flip=None, cls_attr="",
               viewbox=None):
    """Return a vector `<g><svg .../></g>` element that drops `svg_text` into the
    box (bx, by, bw, bh) of the parent document, no rasterization.

    The source is inlined as a nested <svg> viewport (preserveAspectRatio="none"
    so it fills the box exactly) instead of a base64 <image>, which editors
    render as a bitmap. Ids are namespaced first to avoid clashes between
    panels; matplotlib's global `<style>*{...}</style>` (which would otherwise
    leak document-wide once inlined) is dropped; and an optional flip about the
    box centre is applied via the wrapping <g>.

    `viewbox` may be an (x, y, w, h) tuple to override the source's own viewBox
    — pass the tight bounds of the sub-region you actually want to fill the box
    (e.g. matplotlib's `axes_1`) so surrounding figure whitespace is cropped.
    """
    svg_text = re.sub(r'<\?xml[^>]*\?>', '', svg_text)
    svg_text = re.sub(r'<!DOCTYPE[^>]*>', '', svg_text, flags=re.S)
    svg_text = re.sub(r'<style\b[^>]*>.*?</style>', '', svg_text, flags=re.S)
    svg_text = namespace_ids(svg_text.strip(), f'{elem_id.replace("-", "_")}__')
    m = re.search(r'<svg\b[^>]*>', svg_text)
    open_tag = m.group(0)
    if viewbox is not None:
        viewbox = " ".join(f'{v:.4f}' for v in viewbox)
    else:
        vb = re.search(r'viewBox="([^"]+)"', open_tag)
        if vb:
            viewbox = vb.group(1)
        else:
            w = re.search(r'width="([\d.]+)', open_tag).group(1)
            h = re.search(r'height="([\d.]+)', open_tag).group(1)
            viewbox = f'0 0 {w} {h}'
    nested_open = (f'<svg x="{bx:.4f}" y="{by:.4f}" '
                   f'width="{bw:.4f}" height="{bh:.4f}" viewBox="{viewbox}" '
                   f'preserveAspectRatio="none" '
                   f'xmlns="http://www.w3.org/2000/svg" '
                   f'xmlns:xlink="http://www.w3.org/1999/xlink">')
    inner = svg_text[m.end():]
    tf = flip_transform(flip, bx, by, bw, bh) if flip else ""
    tf_attr = f' transform="{tf}"' if tf else ""
    return f'<g id="{elem_id}"{cls_attr}{tf_attr}>{nested_open}{inner}</g>'


def replace_panel(src, elem_id, svg_path, flip):
    """Replace a single <image id=elem_id> in `src` with an inlined svg_path.

    Returns the updated SVG text. Exits with an error if the element is
    missing or the substitution does not apply.
    """
    line = next((l for l in src.splitlines() if f'id="{elem_id}"' in l), None)
    if line is None:
        sys.exit(f"ERROR: {elem_id} not found")
    cls = re.search(r'class="([^"]*)"', line)
    cls_attr = f' class="{cls.group(1)}"' if cls else ""
    bx, by, bw, bh = bbox_of(line)
    svg_text = open(svg_path).read()
    new = inline_svg(svg_text, elem_id, bx, by, bw, bh, flip=flip,
                     cls_attr=cls_attr)
    out = src.replace(line.strip(), new)
    if out == src:
        sys.exit(f"ERROR: replacement for {elem_id} did not apply")
    print(f"{elem_id:22} <- {os.path.basename(svg_path):34} "
          f"box=({bx:.2f},{by:.2f},{bw:.2f},{bh:.2f}) flip={flip}")
    return out


def _path_points(d):
    """Yield absolute (x, y) points visited by an SVG path `d` string.

    Handles the M/L/H/V/C commands (and lowercase relatives) that matplotlib
    emits; enough to get a correct bounding box for axes frames, ticks and bars.
    """
    toks = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)
    i, cx, cy, cmd = 0, 0.0, 0.0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t; i += 1
            if cmd in "Zz":
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "H":
            x = float(toks[i]); i += 1
            cx = cx + x if rel else x
        elif c == "V":
            y = float(toks[i]); i += 1
            cy = cy + y if rel else y
        else:  # M, L, T, and the endpoint of C/S/Q consume x,y pairs
            n = {"C": 3, "S": 2, "Q": 2, "A": 1}.get(c, 1)
            if c == "A":
                # rx ry rot large sweep x y — only the final x,y are points
                vals = list(map(float, toks[i:i + 7])); i += 7
                x, y = vals[5], vals[6]
                cx = cx + x if rel else x
                cy = cy + y if rel else y
            else:
                for _ in range(n):
                    x = float(toks[i]); y = float(toks[i + 1]); i += 2
                    px = cx + x if rel else x
                    py = cy + y if rel else y
                    yield px, py
                    cx, cy = px, py
                continue
        yield cx, cy


def group_bounds(block):
    """Bounding box (x, y, w, h) of an SVG `<g>` block's drawn content.

    Unions all path geometry with text anchor points (translate()), which
    approximates the visual footprint including axis titles/labels.
    """
    xs, ys = [], []
    for d in re.findall(r'\sd="([^"]*)"', block):
        for x, y in _path_points(d):
            xs.append(x); ys.append(y)
    for tx, ty in re.findall(r'translate\(([-\d.]+)\s+([-\d.]+)\)', block):
        xs.append(float(tx)); ys.append(float(ty))
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def svg_aspect(svg_text):
    """Width/height aspect of an svg from its viewBox (fallback width/height)."""
    vb = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', svg_text)
    if vb:
        w, h = float(vb.group(1)), float(vb.group(2))
    else:
        w = float(re.search(r'width="([\d.]+)', svg_text).group(1))
        h = float(re.search(r'height="([\d.]+)', svg_text).group(1))
    return w / h


def _group_span(text, elem_id):
    """(start, end, indent) of a balanced `<g id=elem_id>...</g>`, or exit."""
    m = re.search(rf'([ \t]*)<g id="{re.escape(elem_id)}">', text)
    if m is None:
        sys.exit(f"ERROR: group {elem_id} not found")
    start = m.start()
    depth, end = 0, None
    for tag in re.finditer(r'<g\b|</g>', text[start:]):
        depth += 1 if tag.group() == '<g' else -1
        if depth == 0:
            end = start + tag.end()
            break
    if end is None:
        sys.exit(f"ERROR: unbalanced <g> for {elem_id}")
    return start, end, m.group(1)


def extract_layer(svg_text, layer_id):
    """Standalone svg keeping only one <g id=layer_id> layer.

    Preserves the <svg> wrapper and *all* <defs> blocks (styles, clip-paths,
    markers, glyphs the layer may reference) while dropping every sibling group
    such as the figure background. Coordinates are unchanged, so the result
    embeds at the same box as the full figure. Assumes no transform on the
    dropped wrappers (true for matplotlib's figure_1/axes_1).
    """
    svg_open = re.search(r'<svg\b[^>]*>', svg_text).group(0)
    defs = re.findall(r'<defs>.*?</defs>', svg_text, re.S)
    start, end, _ = _group_span(svg_text, layer_id)
    body = svg_text[start:end].lstrip()
    return f'{svg_open}\n' + "\n".join(defs) + f'\n {body}\n</svg>\n'


def find_inkscape(explicit=None):
    """Locate an inkscape binary (PATH, common macOS/Homebrew spots), or exit."""
    for cand in (explicit, "inkscape",
                 "/Applications/Inkscape.app/Contents/MacOS/inkscape",
                 "/opt/homebrew/bin/inkscape"):
        if not cand:
            continue
        if os.path.isfile(cand) or shutil.which(cand):
            return cand
    sys.exit("ERROR: inkscape not found (pass inkscape=/path/to/inkscape)")


def inkscape_crop_layer(svg_path, layer, inkscape=None, fonts=None,
                        bold_fonts=None):
    """Standalone plain-svg text of `layer`, canvas cropped to its *true* visual
    bounds via Inkscape (`--export-area-drawing`).

    Inkscape measures real text glyph extents, so tick labels / axis titles are
    included — unlike the regex `group_bounds`, which only sees path geometry
    and `translate()` anchors and therefore clips matplotlib's `<text x= y=>`
    labels. The result is rebased so its viewBox is `0 0 W H`; embed it (and read
    that viewBox as the tight source region) instead of a `group_bounds` slice.

    `fonts`/`bold_fonts` are swapped in *before* the crop, so the bounds reflect
    the display font's real metrics. This matters when the substitute is wider
    than matplotlib's Arial (Graphik is): cropping first and swapping after would
    size the viewBox to the Arial extents, and the now-wider labels would spill
    past it and be clipped by the nested viewport. Inkscape must have the display
    font installed for its metrics to apply (it falls back silently otherwise).

    The first Inkscape run in a session can take ~1 min while it builds its font
    cache; subsequent runs are sub-second.
    """
    ink = find_inkscape(inkscape)
    src_text = open(svg_path).read()
    if fonts:
        src_text = sub_fonts(src_text, fonts, bold_mapping=bold_fonts)
    layer_svg = extract_layer(src_text, layer)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.svg")
        out = os.path.join(d, "out.svg")
        open(src, "w").write(layer_svg)
        subprocess.run(
            [ink, src, "--export-area-drawing", "--export-plain-svg",
             f"--export-filename={out}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return open(out).read()


def inkscape_object_bbox(svg_path, elem_id, inkscape=None):
    """True visual bounding box (x, y, w, h) of object `elem_id` in `svg_path`,
    as measured by Inkscape's `--query-id` on the *whole* document.

    Because the full document is loaded, every style, font and def is in effect
    and Inkscape measures real text glyph extents — unlike the regex
    `group_bounds`, which only sees path geometry + `translate()` anchors and so
    under-measures a panel whose right-side labels anchor at their left edge
    (their glyph width is never counted). Use this to size a replacement slot to
    the footprint the original artwork truly occupies (see the genome_coverage
    `d-panel`, whose long model-name row labels overflow the regex bounds).
    """
    ink = find_inkscape(inkscape)
    out = subprocess.run(
        [ink, svg_path, f"--query-id={elem_id}",
         "--query-x", "--query-y", "--query-width", "--query-height"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True).stdout
    x, y, w, h = (float(v) for v in out.split())
    return x, y, w, h


def _family_pattern(src_family):
    """Regex matching a `font-family:` declaration for `src_family`.

    Matches the family quoted ('Arial' / "Arial") or bare (Arial). The bare
    form must NOT swallow a trailing quote — that quote is the closing
    delimiter of the enclosing style="..." attribute, and eating it leaves the
    attribute unterminated (invalid XML). A negative lookahead keeps the match
    to the family token itself.
    """
    name = re.escape(src_family)
    return re.compile(
        rf"font-family\s*:\s*(?:'{name}'|\"{name}\"|{name}(?![\w-]))"
    )


# a bold `font-weight` declaration: keyword `bold`/`bolder` or a numeric weight
# in the 600–900 (semibold and up) range that matplotlib emits (e.g. `700`).
_BOLD_WEIGHT_RE = re.compile(r"font-weight\s*:\s*(?:bold(?:er)?|[6-9]00)")

# glyphs the display face (Graphik) renders poorly or lacks, kept in Arial by
# wrapping each in its own <tspan>. The right-arrow → used in the genome_coverage
# "Required mapping" column is the case that matters.
KEEP_ARIAL_GLYPHS = ("→",)  # →


def sub_fonts(svg_text, mapping, bold_mapping=None,
              keep_arial_glyphs=KEEP_ARIAL_GLYPHS):
    """Swap `font-family` declarations, robust to CSS spacing/quoting.

    `mapping` maps a source family name (e.g. ``"Arial"``) to its replacement
    font-family value (e.g. ``"GraphikTrial-Regular, 'Graphik Trial'"``).

    A regex (rather than a literal `str.replace`) is required because the
    declaration is written differently depending on who last touched the file:
    matplotlib emits ``font-family: 'Arial'`` (space + single quotes) while
    Inkscape's ``--export-plain-svg`` (run by `inkscape_crop_layer` whenever a
    `layer` is extracted) normalizes it to ``font-family:Arial`` (no space, no
    quotes). The old literal mapping only matched the matplotlib form, so the
    swap silently no-op'd on any layer that had been through Inkscape.

    `bold_mapping` (same keys as `mapping`) opts into weight-aware swapping: any
    element whose ``style="..."`` also declares a bold ``font-weight`` (the
    keyword `bold` or a numeric 600–900, e.g. matplotlib's `700`) gets the bold
    replacement family instead of the regular one. This lets a single Arial
    source map to a regular display face and a separate semibold face. The swap
    is applied per `style` attribute — matplotlib and Inkscape both keep
    ``font-family`` and ``font-weight`` together in one ``style`` — so the
    weight of one label never leaks onto its neighbours.

    `keep_arial_glyphs` are characters pinned back to Arial *after* the family
    swap: each occurrence is wrapped in its own ``<tspan font-family:'Arial'>``
    so it renders in Arial while the rest of its label still becomes the display
    font. Graphik has no proper right-arrow, so the → in labels like
    ``gene ID → human ortholog`` would otherwise turn into a fallback box.
    Because these are bare characters inside `<text>` content (never inside a
    tag or attribute in the panels we process), a plain replace is safe.
    """
    if bold_mapping:
        fam_res = {fam: _family_pattern(fam) for fam in mapping}

        def repl_style(m):
            style = m.group(0)
            bold = bool(_BOLD_WEIGHT_RE.search(style))
            for fam, pat in fam_res.items():
                value = bold_mapping.get(fam, mapping[fam]) if bold else mapping[fam]
                style = pat.sub(f"font-family:{value}", style)
            return style

        svg_text = re.sub(r'style="[^"]*"', repl_style, svg_text)
    else:
        for src_family, new_value in mapping.items():
            svg_text = _family_pattern(src_family).sub(
                f"font-family:{new_value}", svg_text)

    for g in keep_arial_glyphs or ():
        svg_text = svg_text.replace(
            g, f"<tspan style=\"font-family: 'Arial'\">{g}</tspan>")
    return svg_text


def load_svg(svg_path, layer=None, fonts=None, bold_fonts=None):
    """Read an svg and optionally keep only `layer` and remap `fonts`."""
    text = open(svg_path).read()
    if layer:
        text = extract_layer(text, layer)
    if fonts:
        text = sub_fonts(text, fonts, bold_mapping=bold_fonts)
    return text


def replace_group(src, elem_id, svg_path, box=None, measure_from=None,
                  layer=None, fonts=None, bold_fonts=None, inkscape=None):
    """Replace a whole `<g id=elem_id>...</g>` block with an inlined svg.

    The source svg is optionally reduced to a single `layer` and has `fonts`
    remapped before embedding.

    Sizing (when `box` is not given verbatim), the target slot `T` is the
    original group's own footprint. By default that footprint is the regex
    `group_bounds`; pass `measure_from` (a path to the svg the group lives in,
    usually the input figure) to instead measure `elem_id`'s *true* glyph-aware
    bounds with Inkscape (see `inkscape_object_bbox`) — needed for panels like
    genome_coverage whose long right-side labels overflow the regex bounds. The
    whole visible source region — the true glyph-aware bounds of `layer` (e.g.
    matplotlib's `axes_1`, tick labels and axis titles included), else the svg's
    own viewBox — is fit *inside* the slot with a single uniform scale (aspect
    ratio preserved, "contain": scale-to-fit, never overflowing the container)
    and centred in it.

    When `layer` is set the source region comes from Inkscape (see
    `inkscape_crop_layer`): the layer is cropped to its real visual bounds and
    rebased to a `0 0 W H` viewBox, so text is never clipped the way the regex
    `group_bounds` would clip it. Without a layer, the svg's own viewBox is used.

    `box` may instead be given as (x, y, w, h) and is then used verbatim.
    The image fills the box exactly (preserveAspectRatio="none").
    """
    start, end, indent = _group_span(src, elem_id)
    block = src[start:end]
    # S: the visible source region, used as both the nested viewBox and the box
    # we contain in the slot.
    if layer:
        # swap fonts *before* the crop so the viewBox reflects the display
        # font's true (wider) glyph extents; otherwise the labels overflow the
        # Arial-measured bounds and get clipped by the nested viewport.
        svg_text = inkscape_crop_layer(svg_path, layer, inkscape=inkscape,
                                       fonts=fonts, bold_fonts=bold_fonts)
    else:
        svg_text = load_svg(svg_path, fonts=fonts, bold_fonts=bold_fonts)
    vb = re.search(r'viewBox="([^"]+)"', svg_text)
    sx0, sy0, sw, sh = map(float, vb.group(1).split())
    src_vb = (sx0, sy0, sw, sh)
    # T: target slot. box verbatim > Inkscape's true glyph-aware bounds of the
    # original group (measure_from) > regex group_bounds.
    if box:
        tx, ty, tw, th = box
    elif measure_from:
        tx, ty, tw, th = inkscape_object_bbox(measure_from, elem_id,
                                               inkscape=inkscape)
    else:
        tx, ty, tw, th = group_bounds(block)
    # contain: uniform scale so the whole region fits inside the slot, centred.
    k = min(tw / sw, th / sh)
    bw, bh = sw * k, sh * k
    bx = tx + (tw - bw) / 2
    by = ty + (th - bh) / 2
    new = indent + inline_svg(svg_text, elem_id, bx, by, bw, bh, viewbox=src_vb)
    print(f"{elem_id:22} <- {os.path.basename(svg_path):34} "
          f"box=({bx:.2f},{by:.2f},{bw:.2f},{bh:.2f}) [group"
          f"{' layer='+layer if layer else ''}]")
    return src[:start] + new + src[end:]


def strip_illustrator_pgf(svg_text):
    """Remove Adobe Illustrator's private PGF artwork copy from an AI-exported svg.

    Illustrator embeds a full, compressed copy of the *original* artwork in an
    `<i:aipgf>` blob (referenced by `<i:aipgfRef>`) inside a top-level
    `<metadata>`. On reopen, Illustrator restores the drawing from this private
    copy and IGNORES the actual SVG markup — so the panel swaps we write into the
    file are invisible in Illustrator, which keeps showing the stale original.
    Dropping the block forces Illustrator to parse the real SVG (and shrinks the
    file, since the blob is tens of MB). Matplotlib's own `<metadata>` RDF blocks
    inside the inlined panels do not contain `i:aipgfRef`, so they're untouched.
    """
    svg_text = re.sub(r'<metadata>\s*<i:aipgfRef\b.*?</metadata>\s*',
                      '', svg_text, flags=re.S)
    svg_text = re.sub(r'<i:aipgfRef\b[^>]*/>\s*', '', svg_text)
    return svg_text


def apply_groups(svg_in, svg_out, groups):
    """Apply an ordered list of job groups to one SVG and return the new text.

    `groups` is a list of (svg_dir, jobs) where jobs maps
    elem_id -> (svg_filename, flip). All groups are applied sequentially to the
    same document, so panels from different figures can share one output file.
    If `svg_out` is given the result is also written there.
    """
    src = open(svg_in).read()
    for svg_dir, jobs in groups:
        for elem_id, (svg_file, flip) in jobs.items():
            src = replace_panel(src, elem_id, os.path.join(svg_dir, svg_file), flip)
    if svg_out:
        open(svg_out, "w").write(src)
        print(f"wrote {svg_out}")
    return src
