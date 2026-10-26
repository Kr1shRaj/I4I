"""
Render the Mermaid flowcharts in docs/SYSTEM_ARCHITECTURE.md to PNG + SVG
=========================================================================
docs/SYSTEM_ARCHITECTURE.md is the single source: every ```mermaid block that follows a marker

    <!-- diagram: 01_macro_architecture | Macro end-to-end architecture -->

is validated and drawn to docs/diagrams/<name>.png (220 dpi) and <name>.svg for Word / Google Docs.
Graphviz and mermaid-cli are not installed on this machine, so layout is done here with matplotlib:
each subgraph is a horizontal band (top to bottom, in source order); inside a band, nodes are placed
in columns by longest-path rank (wrapping after MAX_COLS) and edges are routed orthogonally through
the gaps between rows, or through the side gutters when a straight route would cross a node.

Validation covers the Mermaid subset the document uses (flowchart TD, subgraph/end, 4 node shapes,
-->, -.->, ==> with optional |"label"|, classDef, class, style). It is stricter than Mermaid
(declare nodes before use, quote every label) but it is not mermaid.js itself.

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\generate_architecture_diagrams.py
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\generate_architecture_diagrams.py --check   # validate only
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Layout constants (inches; the figure is drawn 1 data unit = 1 inch)
# ---------------------------------------------------------------------------
FIG_W = 12.0
MARGIN = 0.75          # outer gutter on each side (edge lanes run here)
BAND_PAD = 0.25        # inside a band, left/right of the nodes
COL_GAP = 0.85         # between node columns (edge labels sit here)
MAX_COLS = 5
MAX_NODE_W = 2.3
STACK_GAP = 0.22       # between nodes stacked in one column
TITLE_H = 0.40         # band title strip
BAND_PAD_B = 0.20
ROW_GAP = 0.50         # minimum gap between row groups inside a band
BAND_GAP = 0.50        # minimum gap between bands
TRACK_SP = 0.22        # spacing of parallel horizontal edge tracks in one gap
LANE_SP = 0.13         # spacing of vertical gutter lanes
HEADER_H = 0.80
LEGEND_H = 1.05

FS_NODE, FS_BAND, FS_EDGE, FS_TITLE = 10, 11, 8, 15
CHAR_W = 0.56          # average glyph width in em (DejaVu Sans), used for wrapping
LINE_H = 1.28          # line height in em

BAND_TINTS = ["#F8FAFC", "#F0F9FF", "#F7FEE7", "#FFF7ED", "#FDF4FF", "#ECFEFF", "#FEFCE8"]
DEFAULT_STYLE = {"fill": "#FFFFFF", "stroke": "#334155", "color": "#0F172A", "stroke-width": "1.5px"}
EDGE_STYLE = {
    "-->": dict(color="#475569", lw=1.3, ls="-"),
    "-.->": dict(color="#64748B", lw=1.2, ls=(0, (4, 3))),
    "==>": dict(color="#0F766E", lw=2.6, ls="-"),
}
LEGEND = [("src", "capsule", "Input / source"), ("proc", "rect", "Process"), ("gate", "diamond", "Decision / gate"),
          ("store", "cylinder", "Storage / checkpoint"), ("reject", "capsule", "Rejected / dropped"),
          ("out", "capsule", "Accepted output"), ("planned", "rect", "Proposed (not built)")]

# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------
LBL = r'"([^"]*)"'
NODE_PATTERNS = [
    ("capsule", re.compile(r"^(\w+)\(\[" + LBL + r"\]\)$")),
    ("cylinder", re.compile(r"^(\w+)\[\(" + LBL + r"\)\]$")),
    ("rect", re.compile(r"^(\w+)\[" + LBL + r"\]$")),
    ("diamond", re.compile(r"^(\w+)\{" + LBL + r"\}$")),
]
EDGE_RE = re.compile(r"^(\w+) (-->|-\.->|==>)(?:\|" + LBL + r"\|)? (\w+)$")
SUBGRAPH_RE = re.compile(r"^subgraph (\w+)\[" + LBL + r"\]$")
CLASSDEF_RE = re.compile(r"^classDef (\w+) (.+)$")
CLASS_RE = re.compile(r"^class ([\w,]+) (\w+)$")
STYLE_RE = re.compile(r"^style (\w+) (.+)$")
RESERVED = {"end", "graph", "flowchart", "subgraph", "class", "classdef", "style", "click", "default", "linkstyle"}
BAD_CHARS = re.compile(r"[;#&`]")
MARKER_RE = re.compile(r"<!--\s*diagram:\s*([\w-]+)\s*\|\s*(.*?)\s*-->")


@dataclass
class Node:
    id: str
    shape: str
    text: str
    band: "Band" = None
    cls: str | None = None
    style: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)
    w: float = 0.0
    h: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    rank: int = 0
    rg: int = 0        # global row-group index
    col: int = 0
    top: bool = True   # topmost / bottommost in its column stack
    bottom: bool = True

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.h

    @property
    def cx(self):
        return self.x0 + self.w / 2

    @property
    def cy(self):
        return self.y0 + self.h / 2


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    label: str | None
    route: str = ""
    gaps: list = field(default_factory=list)   # [(gap index, (xa, xb), slot)] -> track filled later
    lane: tuple | None = None                  # (side, (gap_a, gap_b)) -> lane index filled later
    tracks: dict = field(default_factory=dict)
    lane_idx: int = 0


@dataclass
class Band:
    id: str
    title: str
    nodes: list = field(default_factory=list)
    style: dict = field(default_factory=dict)
    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    node_w: float = 0.0
    ncols: int = 1
    rgs: list = field(default_factory=list)


@dataclass
class RowGroup:
    band: Band
    stacks: dict = field(default_factory=dict)   # col -> [Node]
    h: float = 0.0
    y0: float = 0.0
    first: bool = False
    last: bool = False


@dataclass
class Diagram:
    name: str
    title: str
    nodes: dict = field(default_factory=dict)
    bands: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    classdefs: dict = field(default_factory=dict)


def parse_css(s: str) -> dict:
    out = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def check_label(text: str, where: str, errors: list):
    plain = text.replace("<br/>", "")
    if BAD_CHARS.search(plain):
        errors.append(f"{where}: label {text!r} contains one of ; # & ` (special in Mermaid)")
    if "<" in plain or ">" in plain:
        errors.append(f"{where}: label {text!r} contains a bare < or > (only <br/> is allowed)")
    if "|" in plain or "-->" in plain:
        errors.append(f"{where}: label {text!r} contains | or -->")


def parse_mermaid(name: str, title: str, src: str) -> tuple[Diagram, list]:
    d = Diagram(name, title)
    errors: list[str] = []
    band: Band | None = None
    root_n = 0
    header_seen = False
    pending_class: list[tuple[int, list, str]] = []
    pending_style: list[tuple[int, str, dict]] = []

    for ln, raw in enumerate(src.splitlines(), 1):
        line = raw.strip()
        where = f"{name}:{ln}"
        if not line or line.startswith("%%"):
            continue
        if not header_seen:
            if not re.fullmatch(r"flowchart (TD|TB)", line):
                errors.append(f"{where}: first statement must be 'flowchart TD' (got {line!r})")
            header_seen = True
            continue
        if m := SUBGRAPH_RE.match(line):
            if band is not None and not band.id.startswith("_root"):
                errors.append(f"{where}: nested subgraphs are not supported by this renderer")
            if m.group(1).lower() in RESERVED or m.group(1) in d.nodes or any(b.id == m.group(1) for b in d.bands):
                errors.append(f"{where}: subgraph id {m.group(1)!r} is reserved or already used")
            check_label(m.group(2), where, errors)
            band = Band(m.group(1), m.group(2))
            d.bands.append(band)
            continue
        if line == "end":
            if band is None or band.id.startswith("_root"):
                errors.append(f"{where}: 'end' without an open subgraph")
            band = None
            continue
        if m := CLASSDEF_RE.match(line):
            d.classdefs[m.group(1)] = parse_css(m.group(2))
            continue
        if m := CLASS_RE.match(line):
            pending_class.append((ln, m.group(1).split(","), m.group(2)))
            continue
        if m := STYLE_RE.match(line):
            pending_style.append((ln, m.group(1), parse_css(m.group(2))))
            continue
        if m := EDGE_RE.match(line):
            a, kind, label, b = m.groups()
            for nid in (a, b):
                if nid not in d.nodes:
                    errors.append(f"{where}: edge uses undeclared node {nid!r} (declare nodes before edges)")
            if label is not None:
                check_label(label, where, errors)
            d.edges.append(Edge(a, b, kind, label))
            continue
        for shape, pat in NODE_PATTERNS:
            if m := pat.match(line):
                nid, text = m.groups()
                if nid.lower() in RESERVED:
                    errors.append(f"{where}: node id {nid!r} is reserved in Mermaid")
                if nid in d.nodes:
                    errors.append(f"{where}: node {nid!r} declared twice")
                check_label(text, where, errors)
                if band is None:   # top-level nodes get an untitled band of their own
                    root_n += 1
                    band = Band(f"_root{root_n}", "")
                    d.bands.append(band)
                node = Node(nid, shape, text, band)
                d.nodes[nid] = node
                band.nodes.append(node)
                break
        else:
            errors.append(f"{where}: unrecognised statement {line!r}")

    if band is not None and not band.id.startswith("_root"):
        errors.append(f"{name}: subgraph {band.id!r} is never closed with 'end'")
    for ln, ids, cname in pending_class:
        if cname not in d.classdefs:
            errors.append(f"{name}:{ln}: class {cname!r} has no classDef")
        for nid in ids:
            if nid not in d.nodes:
                errors.append(f"{name}:{ln}: class statement names undeclared node {nid!r}")
            else:
                d.nodes[nid].cls = cname
    for ln, target, css in pending_style:
        if target in d.nodes:
            d.nodes[target].style = css
        elif any(b.id == target for b in d.bands):
            next(b for b in d.bands if b.id == target).style = css
        else:
            errors.append(f"{name}:{ln}: style targets unknown id {target!r}")
    if not d.nodes:
        errors.append(f"{name}: no nodes")
    return d, errors


def extract_blocks(md_text: str) -> list[tuple[str, str, str]]:
    """[(name, title, mermaid source)] for every marker followed by a ```mermaid block."""
    blocks, pending = [], None
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if m := MARKER_RE.search(lines[i]):
            pending = (m.group(1), m.group(2))
        elif lines[i].strip() == "```mermaid":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            if pending is None:
                raise SystemExit(f"line {i + 1}: mermaid block without a '<!-- diagram: name | title -->' marker")
            blocks.append((pending[0], pending[1], "\n".join(lines[i + 1:j])))
            pending = None
            i = j
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def wrap(text: str, max_chars: int, split_words: bool = True) -> list[str]:
    """Greedy wrap on spaces; words longer than a line break after / _ - (else .), or stay whole."""
    max_chars = max(6, max_chars)
    out = []
    for para in text.split("<br/>"):
        line = ""
        for word in para.split():
            parts = []
            while split_words and len(word) > max_chars:
                cut = max(word.rfind(c, 0, max_chars) for c in "/_-")
                if cut <= 0:
                    cut = word.rfind(".", 0, max_chars)
                if cut <= 0:
                    break   # no natural break: keep the word whole, the renderer shrinks the line to fit
                cut += 1
                parts.append(word[:cut])
                word = word[cut:]
            parts.append(word)
            for i, part in enumerate(parts):
                sep = "" if (not line or i > 0) else " "   # pieces of one broken word join without a space
                if line and len(line) + len(sep) + len(part) > max_chars:
                    out.append(line)
                    line = part
                else:
                    line += sep + part
        out.append(line)
    return out


def text_w(s: str, fs: float) -> float:
    return len(s) * fs * CHAR_W / 72


def line_h(fs: float) -> float:
    return fs * LINE_H / 72


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def band_ranks(band: Band, d: Diagram):
    ids = {n.id for n in band.nodes}
    order = {n.id: i for i, n in enumerate(band.nodes)}
    succ = {i: [] for i in ids}
    pred = {i: [] for i in ids}
    # drop edges that close a cycle (DFS back edges, in declaration order)
    adj = {i: [e.dst for e in d.edges if e.src == i and e.dst in ids] for i in ids}
    state, keep = {}, set()

    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1:
                continue
            keep.add((u, v))
            if v not in state:
                dfs(v)
        state[u] = 2

    for n in band.nodes:
        if n.id not in state:
            dfs(n.id)
    for u, v in keep:
        succ[u].append(v)
        pred[v].append(u)
    rank = {}

    def r(u):
        if u not in rank:
            rank[u] = 0 if not pred[u] else 1 + max(r(p) for p in pred[u])
        return rank[u]

    for n in band.nodes:
        r(n.id)
    # pull sources right, next to their earliest successor
    for n in band.nodes:
        if not pred[n.id] and succ[n.id]:
            rank[n.id] = max(0, min(rank[s] for s in succ[n.id]) - 1)
    connected = [n for n in band.nodes if pred[n.id] or succ[n.id]]
    isolated = [n for n in band.nodes if not (pred[n.id] or succ[n.id])]
    nxt = (max(rank[n.id] for n in connected) + 1) if connected else 0
    for n in isolated:   # isolated nodes take free columns to the right, else stack in column 0
        if nxt % MAX_COLS != 0 or nxt == 0 or nxt < MAX_COLS:
            rank[n.id] = nxt
            nxt += 1
        else:
            rank[n.id] = 0
    for n in band.nodes:
        n.rank = rank[n.id]
    return pred, order


def size_node(n: Node, w: float):
    n.w = w
    lh = line_h(FS_NODE)
    n.lines = wrap(n.text, _wrap_chars(n))
    if n.shape == "diamond":
        n.h = max(0.85, len(n.lines) * lh * 2.4 + 0.2)
    else:
        n.h = len(n.lines) * lh + 0.24 + (0.22 if n.shape == "cylinder" else 0.0)


def layout(d: Diagram):
    inner_x0, inner_x1 = MARGIN + BAND_PAD, FIG_W - MARGIN - BAND_PAD
    avail = inner_x1 - inner_x0
    rgs: list[RowGroup] = []
    for band in d.bands:
        pred, order = band_ranks(band, d)
        maxrank = max(n.rank for n in band.nodes)
        band.ncols = min(MAX_COLS, maxrank + 1)
        band.node_w = min(MAX_NODE_W, (avail - (band.ncols - 1) * COL_GAP) / band.ncols)
        row_w = band.ncols * band.node_w + (band.ncols - 1) * COL_GAP
        start = inner_x0 + (avail - row_w) / 2
        band.x0, band.x1 = MARGIN, FIG_W - MARGIN
        n_local = maxrank // MAX_COLS + 1
        local = [RowGroup(band) for _ in range(n_local)]
        for n in band.nodes:
            size_node(n, band.node_w)
            n.col = n.rank % MAX_COLS
            n.x0 = start + n.col * (band.node_w + COL_GAP)
            local[n.rank // MAX_COLS].stacks.setdefault(n.col, []).append(n)
        local[0].first, local[-1].last = True, True
        for k, rg in enumerate(local):
            # order each stack by the mean slot of its predecessors (one barycentre pass)
            for col in sorted(rg.stacks):
                stack = rg.stacks[col]
                prev = rg.stacks.get(col - 1, [])
                pos = {p.id: i for i, p in enumerate(prev)}

                def key(n):
                    ps = [pos[p] for p in pred[n.id] if p in pos]
                    return (sum(ps) / len(ps) if ps else 1e9, order[n.id])

                stack.sort(key=key)
                for i, n in enumerate(stack):
                    n.top, n.bottom = i == 0, i == len(stack) - 1
                    n.rg = len(rgs)
            rg.h = max(max(sum(n.h for n in s) + STACK_GAP * (len(s) - 1) for s in rg.stacks.values()), 0.5)
            band.rgs.append(len(rgs))
            rgs.append(rg)
    return rgs


def exit_x(n: Node, band: Band) -> float:
    return n.cx if n.bottom else min(n.x1 + COL_GAP / 2, band.x1 - 0.1)


def entry_x(n: Node, band: Band) -> float:
    return n.cx if n.top else max(n.x0 - COL_GAP / 2, band.x0 + 0.1)


def under_x(n: Node, band: Band) -> float:
    return n.cx if n.bottom else min(n.x1 + COL_GAP / 2, band.x1 - 0.1)


def column_free(x: float, rg_ids, rgs, d: Diagram) -> bool:
    for j in rg_ids:
        for stack in rgs[j].stacks.values():
            for n in stack:
                if n.x0 - 0.06 <= x <= n.x1 + 0.06:
                    return False
    return True


def plan_routes(d: Diagram, rgs):
    gR, gL = FIG_W - MARGIN, MARGIN   # gutter reference lines (lanes are offset outward)
    for e in d.edges:
        s, t = d.nodes[e.src], d.nodes[e.dst]
        bs, bt = s.band, t.band
        if s.rg == t.rg and t.col == s.col + 1:
            e.route = "A"
        elif s.rg == t.rg:
            e.route = "U"
            e.gaps = [(s.rg + 1, (exit_x(s, bs), under_x(t, bt)))]
        elif t.rg > s.rg:
            xs, xe = exit_x(s, bs), entry_x(t, bt)
            mids = range(s.rg + 1, t.rg)
            if column_free(xe, mids, rgs, d):
                e.route, e.gaps = "B1", [(s.rg + 1, (xs, xe))]
            elif column_free(xs, mids, rgs, d):
                e.route, e.gaps = "B2", [(t.rg, (xs, xe))]
            else:
                e.route = "GR"
                e.gaps = [(s.rg + 1, (xs, gR)), (t.rg, (xe, gR))]
                e.lane = ("R", (s.rg + 1, t.rg))
        else:
            e.route = "GL"
            xs, xe = exit_x(s, bs), entry_x(t, bt)
            e.gaps = [(s.rg + 1, (gL, xs)), (t.rg, (gL, xe))]
            e.lane = ("L", (t.rg, s.rg + 1))
    # tracks: interval colouring per gap
    used: dict[int, list] = {}
    for e in d.edges:
        for g, (xa, xb) in e.gaps:
            lo, hi = min(xa, xb) - 0.12, max(xa, xb) + 0.12
            tracks = used.setdefault(g, [])
            k = 0
            while any(tk == k and not (hi < a or lo > b) for tk, a, b in tracks):
                k += 1
            tracks.append((k, lo, hi))
            e.tracks[g] = k
    ntracks = {g: 1 + max(k for k, _, _ in v) for g, v in used.items()}
    # gutter lanes: interval colouring over gap indices
    lanes: dict[str, list] = {"L": [], "R": []}
    for e in d.edges:
        if e.lane:
            side, (a, b) = e.lane
            k = 0
            while any(lk == k and not (b < la or a > lb) for lk, la, lb in lanes[side]):
                k += 1
            lanes[side].append((k, a, b))
            e.lane_idx = k
    nlanes = {s: (1 + max(k for k, _, _ in v)) if v else 0 for s, v in lanes.items()}
    return ntracks, nlanes


def place_vertically(d: Diagram, rgs, ntracks):
    R = len(rgs)
    gap_top, gap_h = {}, {}
    y = HEADER_H
    for j in range(R + 1):
        base = (0.30 if j in (0, R) else (BAND_GAP if rgs[j].first else ROW_GAP))
        gap_h[j] = max(base, (ntracks.get(j, 0) + 1) * TRACK_SP)
        gap_top[j] = y
        y += gap_h[j]
        if j == R:
            break
        rg = rgs[j]
        if rg.first:
            rg.band.y0 = y
            y += TITLE_H if rg.band.title else 0.12
        rg.y0 = y
        for stack in rg.stacks.values():
            total = sum(n.h for n in stack) + STACK_GAP * (len(stack) - 1)
            yy = rg.y0 + (rg.h - total) / 2
            for n in stack:
                n.y0 = yy
                yy += n.h + STACK_GAP
        y += rg.h
        if rg.last:
            y += BAND_PAD_B
            rg.band.y1 = y
    return gap_top, gap_h, y + LEGEND_H


# ---------------------------------------------------------------------------
# Edge geometry
# ---------------------------------------------------------------------------
def simplify(pts):
    out = []
    for p in pts:
        if out and abs(p[0] - out[-1][0]) < 1e-6 and abs(p[1] - out[-1][1]) < 1e-6:
            continue
        out.append(p)
    i = 1
    while i < len(out) - 1:
        (ax, ay), (bx, by), (cx, cy) = out[i - 1], out[i], out[i + 1]
        if (abs(ax - bx) < 1e-6 and abs(bx - cx) < 1e-6) or (abs(ay - by) < 1e-6 and abs(by - cy) < 1e-6):
            out.pop(i)
        else:
            i += 1
    return out


def edge_points(e: Edge, d: Diagram, gy, gutter_x):
    s, t = d.nodes[e.src], d.nodes[e.dst]
    bs, bt = s.band, t.band

    def exit_down(y):
        if s.bottom:
            return [(s.cx, s.y1), (s.cx, y)]
        x = exit_x(s, bs)
        return [(s.x1, s.cy), (x, s.cy), (x, y)]

    def enter_top(y):
        if t.top:
            return [(t.cx, y), (t.cx, t.y0)]
        x = entry_x(t, bt)
        return [(x, y), (x, t.cy), (t.x0, t.cy)]

    def enter_bottom(y):
        if t.bottom:
            return [(t.cx, y), (t.cx, t.y1)]
        x = under_x(t, bt)
        return [(x, y), (x, t.cy), (t.x1, t.cy)]

    if e.route == "A":
        mx = s.x1 + COL_GAP / 2
        return simplify([(s.x1, s.cy), (mx, s.cy), (mx, t.cy), (t.x0, t.cy)])
    if e.route == "U":
        g = e.gaps[0][0]
        return simplify(exit_down(gy(g, e)) + enter_bottom(gy(g, e)))
    if e.route in ("B1", "B2"):
        g = e.gaps[0][0]
        return simplify(exit_down(gy(g, e)) + enter_top(gy(g, e)))
    (g1, _), (g2, _) = e.gaps
    x = gutter_x(e)
    return simplify(exit_down(gy(g1, e)) + [(x, gy(g1, e)), (x, gy(g2, e))] + enter_top(gy(g2, e)))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def node_style(n: Node, d: Diagram) -> dict:
    st = dict(DEFAULT_STYLE)
    if n.cls and n.cls in d.classdefs:
        st.update(d.classdefs[n.cls])
    st.update(n.style)
    return st


def dash(st: dict):
    if "stroke-dasharray" in st:
        nums = [float(v) for v in re.split(r"[\s,]+", st["stroke-dasharray"]) if v]
        return (0, tuple(nums))
    return "-"


def px(v: str, default=1.5) -> float:
    try:
        return float(str(v).replace("px", ""))
    except ValueError:
        return default


def draw_shape(ax, shape, x0, y0, w, h, st, z=3):
    fc, ec = st["fill"], st["stroke"]
    lw, ls = px(st.get("stroke-width", "1.5px")) * 0.9, dash(st)
    if shape == "rect":
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    elif shape == "capsule":
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={min(h, w) / 2:.3f}",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    elif shape == "diamond":
        ax.add_patch(Polygon([(x0, y0 + h / 2), (x0 + w / 2, y0), (x0 + w, y0 + h / 2), (x0 + w / 2, y0 + h)],
                             closed=True, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, joinstyle="round"))
    elif shape == "cylinder":
        e = min(0.18, h * 0.3)
        ax.add_patch(Ellipse((x0 + w / 2, y0 + h - e / 2), w, e, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
        ax.add_patch(Rectangle((x0, y0 + e / 2), w, h - e, fc=fc, ec="none", zorder=z))
        ax.plot([x0, x0], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
        ax.plot([x0 + w, x0 + w], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=lw, ls=ls, zorder=z)
        ax.add_patch(Ellipse((x0 + w / 2, y0 + e / 2), w, e, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def draw_node(ax, n: Node, d: Diagram, renderer):
    st = node_style(n, d)
    draw_shape(ax, n.shape, n.x0, n.y0, n.w, n.h, st)
    lh = line_h(FS_NODE)
    title_lines = len(wrap(n.text.split("<br/>")[0], _wrap_chars(n)))   # the first <br/> segment is bold
    cy = n.cy + (0.05 if n.shape == "cylinder" else 0.0)
    dpi = ax.figure.dpi
    for i, line in enumerate(n.lines):
        y = cy - (len(n.lines) - 1) / 2 * lh + i * lh
        fs = FS_NODE if i < title_lines else FS_NODE - 1
        t = ax.text(n.cx, y, line, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if i < title_lines else "normal", color=st.get("color", "#0F172A"), zorder=4)
        # shrink a line that is wider than the shape allows at its height (diamonds narrow towards the tips)
        if n.shape == "diamond":
            half = abs(y - n.cy) + lh / 2
            avail = n.w * max(0.2, 1 - half / (n.h / 2)) - 0.08
        else:
            avail = n.w - (0.30 if n.shape == "capsule" else 0.12)
        width = t.get_window_extent(renderer).width / dpi
        if width > avail:
            t.set_fontsize(max(6.0, fs * avail / width))


def _wrap_chars(n: Node) -> int:
    if n.shape == "diamond":
        return int(0.64 * n.w * 72 / (FS_NODE * CHAR_W))
    inset = 0.34 if n.shape == "capsule" else 0.18
    return int((n.w - inset) * 72 / (FS_NODE * CHAR_W))


def label_box(text_lines, x, y):
    w = max(text_w(s, FS_EDGE) for s in text_lines) + 0.14
    h = len(text_lines) * line_h(FS_EDGE) + 0.10
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def overlaps(a, b, pad=0.02):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def render(d: Diagram, out_dir: Path) -> list[Path]:
    rgs = layout(d)
    ntracks, nlanes = plan_routes(d, rgs)
    gap_top, gap_h, H = place_vertically(d, rgs, ntracks)

    def gy(g, e):
        return gap_top[g] + (e.tracks[g] + 1) * gap_h[g] / (ntracks.get(g, 1) + 1)

    lane_sp = {s: min(LANE_SP, (MARGIN - 0.12) / max(1, n)) for s, n in nlanes.items()}

    def gutter_x(e):
        side = e.lane[0]
        off = lane_sp[side] * (e.lane_idx + 1)
        return FIG_W - MARGIN + off if side == "R" else MARGIN - off

    fig = plt.figure(figsize=(FIG_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(MARGIN, 0.38, d.title, fontsize=FS_TITLE, fontweight="bold", color="#0F172A", va="center")
    ax.text(MARGIN, 0.64, f"Krish 9-class YOLO26-seg  ·  source: docs/SYSTEM_ARCHITECTURE.md  ·  {d.name}",
            fontsize=8.5, color="#64748B", va="center")

    for i, band in enumerate(d.bands):
        if not band.title:
            continue
        fc = band.style.get("fill", BAND_TINTS[i % len(BAND_TINTS)])
        ec = band.style.get("stroke", "#CBD5E1")
        ax.add_patch(FancyBboxPatch((band.x0, band.y0), band.x1 - band.x0, band.y1 - band.y0,
                                    boxstyle="round,pad=0,rounding_size=0.10", fc=fc, ec=ec, lw=1.0, zorder=0))
        ax.text(band.x0 + 0.16, band.y0 + TITLE_H / 2 + 0.01, band.title, fontsize=FS_BAND, fontweight="bold",
                color="#1E293B", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc=fc, ec="none"))

    node_boxes = [(n.x0, n.y0, n.x1, n.y1) for n in d.nodes.values()]
    placed: list[tuple] = []
    a_out = {}
    for e in d.edges:
        if e.route == "A":
            a_out[e.src] = a_out.get(e.src, 0) + 1

    for e in d.edges:
        pts = edge_points(e, d, gy, gutter_x)
        style = EDGE_STYLE[e.kind]
        path = MPath(pts, [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                     mutation_scale=13 if e.kind != "==>" else 15, color=style["color"],
                                     lw=style["lw"], linestyle=style["ls"], shrinkA=0, shrinkB=0,
                                     joinstyle="miter", zorder=2))
        if not e.label:
            continue
        s, t = d.nodes[e.src], d.nodes[e.dst]
        rotation = 0
        if e.route == "A":
            lines = wrap(e.label, int((COL_GAP - 0.1) * 72 / (FS_EDGE * CHAR_W)), split_words=False)
            mx = s.x1 + COL_GAP / 2
            cands = [(mx, t.cy if a_out.get(e.src, 0) > 1 else s.cy)]
            cands += [(mx, cands[0][1] + dy) for dy in (-0.22, 0.22, -0.4, 0.4)]
        elif e.route in ("GR", "GL"):
            lines = [e.label] if text_w(e.label, FS_EDGE) < 2.6 else wrap(e.label, 40)
            x = pts[0][0]
            seg = max(zip(pts, pts[1:]), key=lambda ab: abs(ab[1][1] - ab[0][1]) if abs(ab[0][0] - ab[1][0]) < 1e-6
                      and abs(ab[0][0] - gutter_x(e)) < 1e-6 else -1)
            x, y = seg[0][0], (seg[0][1] + seg[1][1]) / 2
            rotation = 90
            cands = [(x, y + dy) for dy in (0, -0.6, 0.6, -1.2, 1.2)]
        else:
            lines = wrap(e.label, 26, split_words=False)
            horiz = [(a, b) for a, b in zip(pts, pts[1:]) if abs(a[1] - b[1]) < 1e-6 and abs(a[0] - b[0]) > 0.3]
            if horiz:
                a, b = max(horiz, key=lambda ab: abs(ab[0][0] - ab[1][0]))
                y, xa, xb = a[1], min(a[0], b[0]), max(a[0], b[0])
                mid = (xa + xb) / 2
                cands = [(mid, y)] + [(min(max(mid + dx, xa + 0.3), xb - 0.3), y)
                                      for dx in (-0.6, 0.6, -1.2, 1.2, -2.0, 2.0)]
            else:
                vert = max(zip(pts, pts[1:]), key=lambda ab: abs(ab[0][1] - ab[1][1]))
                x, ya, yb = vert[0][0], min(vert[0][1], vert[1][1]), max(vert[0][1], vert[1][1])
                # walk the whole segment from its middle outward, so a label never lands on a node it passes
                steps = [i * 0.15 for i in range(int((yb - ya) / 0.3) + 1)]
                cands = [((x, (ya + yb) / 2 + sgn * dy)) for dy in steps for sgn in (1, -1)
                         if ya + 0.15 <= (ya + yb) / 2 + sgn * dy <= yb - 0.15] or [(x, (ya + yb) / 2)]
        chosen = None
        for (x, y) in cands:
            box = label_box(lines, x, y)
            if rotation:
                w, h = box[2] - box[0], box[3] - box[1]
                box = (x - h / 2, y - w / 2, x + h / 2, y + w / 2)
            if not any(overlaps(box, b) for b in node_boxes + placed):
                chosen = (x, y, box)
                break
        if chosen is None:
            x, y = cands[0]
            box = label_box(lines, x, y)
            chosen = (x, y, box)
        x, y, box = chosen
        placed.append(box)
        ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=FS_EDGE, color="#1E293B",
                rotation=rotation, linespacing=1.15, zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=style["color"], lw=0.6))

    renderer = fig.canvas.get_renderer()
    for n in d.nodes.values():
        draw_node(ax, n, d, renderer)

    # legend: node classes on the first row, edge styles on the second
    ly = H - LEGEND_H + 0.12
    used = {n.cls for n in d.nodes.values()}
    x = MARGIN
    for cls, shape, text in LEGEND:
        if cls not in used or cls not in d.classdefs:
            continue
        st = dict(DEFAULT_STYLE)
        st.update(d.classdefs[cls])
        if x + 0.46 + text_w(text, 8) > FIG_W - 0.2:   # wrap the legend instead of running off the page
            x, ly = MARGIN, ly + 0.34
        draw_shape(ax, shape, x, ly - 0.11, 0.38, 0.22, st)
        ax.text(x + 0.46, ly, text, fontsize=8, va="center", color="#334155")
        x += 0.46 + text_w(text, 8) + 0.22
    ly += 0.34
    x = MARGIN
    for kind, text in (("-->", "flow"), ("==>", "main data path"), ("-.->", "reference / control")):
        if not any(e.kind == kind for e in d.edges):
            continue
        s = EDGE_STYLE[kind]
        ax.add_patch(FancyArrowPatch((x, ly), (x + 0.45, ly), arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                     mutation_scale=12, color=s["color"], lw=s["lw"], linestyle=s["ls"]))
        ax.text(x + 0.53, ly, text, fontsize=8, va="center", color="#334155")
        x += 0.53 + text_w(text, 8) + 0.3

    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [out_dir / f"{d.name}.png", out_dir / f"{d.name}.svg"]
    fig.savefig(outs[0], dpi=220, facecolor="white")
    fig.savefig(outs[1], facecolor="white")
    plt.close(fig)
    return outs


def main():
    ap = argparse.ArgumentParser(description="Render the Mermaid diagrams in SYSTEM_ARCHITECTURE.md")
    ap.add_argument("--doc", type=Path, default=PROJECT_ROOT / "docs" / "SYSTEM_ARCHITECTURE.md")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "diagrams")
    ap.add_argument("--check", action="store_true", help="validate only, draw nothing")
    ap.add_argument("--only", nargs="*", help="diagram names to render (default: all)")
    args = ap.parse_args()

    blocks = extract_blocks(args.doc.read_text(encoding="utf-8"))
    if not blocks:
        raise SystemExit(f"no marked mermaid blocks in {args.doc}")
    diagrams, all_errors = [], []
    for name, title, src in blocks:
        d, errors = parse_mermaid(name, title, src)
        all_errors += errors
        diagrams.append(d)
        print(f"{name}: {len(d.nodes)} nodes, {len(d.edges)} edges, {len(d.bands)} subgraphs"
              + (f", {len(errors)} error(s)" if errors else ", OK"))
    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)
        raise SystemExit(1)
    if args.check:
        return
    for d in diagrams:
        if args.only and d.name not in args.only:
            continue
        for p in render(d, args.out):
            print(f"  wrote {p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p}")


if __name__ == "__main__":
    main()
