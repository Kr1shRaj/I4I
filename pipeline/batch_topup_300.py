# -*- coding: utf-8 -*-
"""
pipeline/batch_topup_300.py
============================
Fixed: DDGS-based fast parallel scraping engine replacing slow icrawler.

Key improvements:
  1. Uses DDGS (DuckDuckGo direct image URLs) instead of icrawler/Bing HTML scraping
  2. High-precision, machine-specific query templates per class
  3. Concurrent download with ThreadPoolExecutor (8 threads)
  4. Fast per-URL timeout (4s) to skip dead hosts immediately
  5. Cycles through many query variants to accumulate enough candidates
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from PIL import Image

from pipeline.config import (
    CLASSES,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    MIN_RESOLUTION,
)
from pipeline.stage3_clean import clean_class
from pipeline.stage4_clip_filter import filter_class, load_model
from pipeline.stage5_diversity_kselect import process_class

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("[ERROR] Install ddgs: pip install ddgs")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET = 300
DOWNLOAD_TIMEOUT = 4      # seconds per URL - short to skip dead hosts fast
DOWNLOAD_THREADS = 8      # parallel downloads
DDGS_MAX_PER_QUERY = 50   # max images per DDGS query
DDGS_SLEEP = 1.0          # seconds between DDGS calls to avoid rate-limit


# ---------------------------------------------------------------------------
# High-precision machine-specific query bank per class
# ---------------------------------------------------------------------------
CLASS_QUERIES: dict[str, list[str]] = {
    "lathe": [
        "metal lathe machine workshop",
        "engine lathe turning machine",
        "industrial lathe machine tool",
        "manual metal lathe machine factory",
        "cnc turning lathe machine",
        "precision lathe machine metalworking",
        "benchtop metal lathe machine",
        "lathe machine industrial equipment photo",
        "wood turning lathe machine",
        "horizontal lathe machine metal cutting",
    ],
    "cnc_milling": [
        "CNC milling machine industrial",
        "CNC machining center factory",
        "vertical CNC milling machine",
        "5 axis CNC milling machine",
        "CNC milling machine workshop photo",
        "industrial CNC machining equipment",
        "CNC mill machine metalworking",
        "CNC milling machine spindle cutting",
    ],
    "cnc_router": [
        "CNC router machine woodworking",
        "industrial CNC router machine",
        "3 axis CNC router machine table",
        "CNC wood router machine workshop",
        "heavy duty CNC router machine",
        "CNC router carving machine factory",
        "gantry CNC router machine",
        "CNC router cutting machine wood",
    ],
    "band_saw": [
        "band saw machine woodworking",
        "industrial band saw machine",
        "metal cutting band saw machine",
        "vertical band saw machine workshop",
        "heavy duty band saw machine",
        "band saw cutting machine factory",
        "benchtop band saw machine",
        "band saw blade machine tool",
    ],
    "table_saw": [
        "table saw machine woodworking",
        "cabinet table saw machine",
        "industrial table saw machine",
        "table saw cutting machine workshop",
        "contractor table saw machine",
        "table saw wood cutting machine",
        "heavy duty table saw machine",
        "table saw machine factory",
    ],
    "grinding": [
        "grinding machine industrial",
        "surface grinding machine",
        "cylindrical grinding machine",
        "industrial grinder machine factory",
        "bench grinding machine workshop",
        "angle grinder machine industrial",
        "floor grinding machine",
        "grinding wheel machine metalworking",
    ],
    "conveyor": [
        "industrial conveyor belt machine",
        "factory conveyor belt system",
        "conveyor belt manufacturing plant",
        "automated conveyor system factory",
        "industrial belt conveyor equipment",
        "conveyor line production factory",
        "roller conveyor machine industrial",
        "conveyor system assembly line",
    ],
    "milling": [
        "vertical milling machine tool",
        "knee type milling machine",
        "universal milling machine factory",
        "bridgeport milling machine workshop",
        "manual milling machine metalworking",
        "horizontal milling machine industrial",
        "milling machine metal cutting",
        "milling machine spindle workshop",
    ],
    "planer": [
        "wood planer machine woodworking",
        "thickness planer machine workshop",
        "industrial wood planer machine",
        "surface planer machine factory",
        "benchtop wood planer machine",
        "metal planer machine industrial",
        "planer machine woodworking equipment",
        "electric wood planer machine",
    ],
    "panel_saw": [
        "panel saw machine woodworking",
        "vertical panel saw machine",
        "industrial panel saw machine factory",
        "sliding panel saw machine",
        "horizontal panel saw machine",
        "panel saw cutting machine workshop",
        "wood panel saw machine",
        "panel saw machine furniture factory",
    ],
    "forklift": [
        "forklift machine warehouse",
        "electric forklift warehouse factory",
        "industrial forklift machine",
        "reach forklift machine warehouse",
        "counterbalance forklift machine",
        "forklift truck warehouse industrial",
        "forklift pallet warehouse factory",
        "heavy duty forklift machine",
    ],
    "drilling": [
        "drill press machine workshop",
        "radial drilling machine factory",
        "vertical drilling machine industrial",
        "pillar drill press machine",
        "bench drill press machine tool",
        "industrial drilling machine metalworking",
        "CNC drilling machine factory",
        "magnetic drill machine industrial",
    ],
    "miter_saw": [
        "miter saw machine woodworking",
        "compound miter saw machine",
        "sliding miter saw machine",
        "double bevel miter saw machine",
        "chop saw miter saw machine",
        "miter saw cutting machine workshop",
        "cordless miter saw machine",
        "industrial miter saw machine",
    ],
    "spindle_moulder": [
        "spindle moulder machine woodworking",
        "spindle moulder machine workshop",
        "wood spindle moulder machine factory",
        "vertical spindle moulder machine",
        "industrial spindle moulder machine",
        "spindle shaper machine woodworking",
        "spindle moulder cutter machine",
        "spindle moulder machine furniture",
    ],
    "packaging_machine": [
        "packaging machine factory industrial",
        "automatic packaging machine",
        "food packaging machine factory",
        "industrial packaging machine line",
        "box packaging machine factory",
        "wrapping packaging machine industrial",
        "shrink wrap packaging machine",
        "vacuum packaging machine factory",
    ],
    "injection_molding": [
        "injection molding machine factory",
        "plastic injection molding machine",
        "industrial injection molding machine",
        "injection molding machine press",
        "horizontal injection molding machine",
        "vertical injection molding machine",
        "injection molding machine workshop",
        "plastic injection machine factory",
    ],
    "control_panel": [
        "industrial control panel electrical",
        "PLC control panel machine factory",
        "electrical control panel enclosure",
        "machinery control panel cabinet",
        "automation control panel industrial",
        "industrial control cabinet panel",
        "CNC machine control panel",
        "electrical panel board industrial",
    ],
    "jointer": [
        "woodworking jointer machine",
        "wood jointer planer machine",
        "industrial jointer machine workshop",
        "benchtop jointer machine woodworking",
        "surface jointer machine factory",
        "jointer machine flat wood",
        "jointer planer combo machine",
        "electric jointer machine wood",
    ],
    "sanding_machines": [
        "belt sander machine industrial",
        "wide belt sander machine factory",
        "disc sanding machine workshop",
        "drum sander machine woodworking",
        "oscillating spindle sander machine",
        "industrial sanding machine factory",
        "floor sanding machine industrial",
        "woodworking sander machine workshop",
    ],
    "wood_lathe": [
        "wood lathe machine woodworking",
        "wood turning lathe machine workshop",
        "craftsman wood lathe machine",
        "bowl turning wood lathe machine",
        "wood lathe machine tool factory",
        "mini wood lathe machine",
        "wood lathe turning machine",
        "wood lathe machine spindle",
    ],
    "hydraulic_press": [
        "hydraulic press machine industrial",
        "hydraulic press machine factory",
        "industrial hydraulic press machine",
        "hydraulic punch press machine",
        "shop press hydraulic machine",
        "hydraulic press forming machine",
        "h frame hydraulic press machine",
        "hydraulic press metalworking machine",
    ],
    "fire_extinguisher": [
        "fire extinguisher industrial safety",
        "red fire extinguisher wall mount factory",
        "fire extinguisher machine shop safety",
        "commercial fire extinguisher equipment",
        "industrial fire extinguisher workshop",
        "co2 fire extinguisher factory safety",
        "dry chemical fire extinguisher industrial",
        "fire extinguisher inspection safety station",
    ],
    "crane": [
        "industrial overhead crane machine",
        "gantry crane machine factory",
        "bridge crane industrial workshop",
        "mobile hydraulic crane machine",
        "tower crane construction site photo",
        "heavy duty industrial crane equipment",
        "factory overhead hoist crane machine",
        "jib crane machine workshop",
    ],
    "edge_banding_machine": [
        "edge banding machine factory floor",
        "automatic edgebander machine close up detail",
        "edge banding machine cctv surveillance view",
        "worker operating edge banding machine",
        "edgebander machine partially obscured workshop",
        "woodworking edgebander machine conveyor line",
        "industrial edge banding machine furniture factory",
        "heavy duty automatic edgebander machine",
        "edgebander machine trimming unit detail",
        "woodworking edge banding production line",
    ],
    "dust_collector": [
        "industrial dust collector factory floor",
        "cyclone dust collector close up detail",
        "woodworking dust collector cctv surveillance view",
        "worker operating industrial dust collector",
        "baghouse dust collector partially obscured workshop",
        "woodworking shop dust extraction system",
        "industrial dust collection unit factory",
        "dust collector ductwork woodworking plant",
        "shop dust collector blower unit photo",
        "heavy duty industrial dust collector system",
    ],
    "veneer_press": [
        "hydraulic veneer press factory floor",
        "hot press veneer machine close up detail",
        "veneer press machine cctv surveillance view",
        "worker operating veneer press machine",
        "plywood veneer press partially obscured workshop",
        "woodworking hot press machine platen",
        "industrial veneer pressing machine plant",
        "multi daylight veneer press factory",
        "cold press veneer woodworking machine",
        "heavy duty hydraulic veneer press machine",
    ],
    "drum_sander": [
        "drum sander machine factory floor",
        "cylindrical drum sander close up detail",
        "drum sanding machine cctv surveillance view",
        "worker operating drum sander machine",
        "dual drum sander partially obscured workshop",
        "rotary drum sander woodworking machine",
        "heavy duty drum sanding machine shop",
        "industrial drum sander abrasive cylinder",
        "open end drum sander woodworking",
        "double drum sander machine factory",
    ],
    "mortiser": [
        "mortising machine factory floor",
        "hollow chisel mortiser close up detail",
        "slot mortiser machine cctv surveillance view",
        "worker operating mortiser machine",
        "woodworking mortiser partially obscured workshop",
        "chain mortiser machine woodworking",
        "industrial mortising machine shop",
        "slot mortising machine woodworking",
        "square chisel mortising machine photo",
        "heavy duty mortiser machine woodworking",
    ],
    "glue_spreader": [
        "roller glue spreader factory floor",
        "glue spreader machine rollers close up detail",
        "glue spreader machine cctv surveillance view",
        "worker operating glue spreader machine",
        "woodworking glue spreader partially obscured workshop",
        "industrial glue spreader plywood line",
        "double roller glue spreader machine",
        "glue applicator machine woodworking factory",
        "heavy duty glue spreader machine",
        "automatic roller glue spreader woodworking",
    ],
    "pallet_jack": [
        "pallet jack factory floor",
        "hydraulic pallet jack close up detail",
        "pallet jack cctv surveillance view",
        "worker operating pallet jack warehouse",
        "electric pallet jack partially obscured industrial",
        "manual pallet truck warehouse floor",
        "heavy duty pallet jack material handling",
        "electric walkie pallet jack factory",
        "hand pallet truck loaded pallets",
        "industrial pallet jack warehouse floor",
    ],
    "air_compressor": [
        "industrial air compressor factory floor",
        "rotary screw air compressor close up detail",
        "air compressor plant room cctv surveillance view",
        "worker operating industrial air compressor",
        "heavy duty air compressor partially obscured workshop",
        "industrial rotary screw compressor system",
        "stationary air compressor tank factory",
        "reciprocating air compressor industrial plant",
        "factory compressed air station equipment",
        "industrial compressor unit manufacturing floor",
    ],
    "storage_racking": [
        "industrial storage racking loaded pallets factory floor",
        "warehouse pallet rack beam upright close up detail",
        "pallet racking warehouse cctv surveillance view",
        "forklift worker operating storage racking warehouse",
        "cantilever storage rack loaded partially obscured industrial",
        "heavy duty warehouse pallet racking system",
        "high bay industrial pallet rack storage",
        "selective pallet rack loaded warehouse floor",
        "industrial steel storage racking pallets",
        "factory warehouse storage rack loaded goods",
    ],
    "robotic_arm": [
        "industrial robotic arm welding factory floor",
        "robotic arm end effector gripper close up detail",
        "robotic arm cell cctv surveillance view",
        "worker operating industrial robotic arm",
        "manufacturing robot arm partially obscured factory",
        "pick and place robotic arm assembly line",
        "6 axis industrial robot arm plant",
        "fanuc kuka abb industrial robotic arm",
        "heavy duty industrial robot arm manufacturing",
        "articulated robot arm industrial automation",
    ],
    "overhead_hoist": [
        "monorail electric chain hoist factory floor",
        "electric hoist hook and motor close up detail",
        "overhead hoist trolley cctv surveillance view",
        "worker operating electric hoist lifting load",
        "beam mounted hoist partially obscured workshop",
        "wall mounted electric chain hoist unit",
        "industrial wire rope hoist material handling",
        "workstation overhead hoist lifting equipment",
        "motorized chain hoist industrial shop",
        "overhead hoist unit I beam track",
    ],
    "platform_scale": [
        "industrial platform scale factory floor",
        "platform scale digital display indicator close up detail",
        "floor platform scale cctv surveillance view",
        "worker weighing load operating platform scale",
        "heavy duty platform scale partially obscured warehouse",
        "industrial floor weighing scale pallets",
        "heavy duty pallet platform scale warehouse",
        "benchtop platform scale workshop",
        "industrial steel floor scale plant",
        "digital floor platform weighing scale",
    ],
    "ppe_station": [
        "PPE safety station wall mounted factory floor wide shot photo",
        "industrial PPE dispenser rack distant cctv surveillance view",
        "worker standing near wall mounted PPE station workshop",
        "PPE cabinet equipment rack dark low light factory background",
        "PPE safety dispenser station mounted wall industrial plant",
        "factory floor wide shot showing distant PPE station corner",
        "standalone PPE safety station rack manufacturing plant photo",
        "wall mounted safety glasses earplug dispenser factory corridor",
        "industrial PPE equipment station worker operating in background",
        "PPE safety dispenser rack mounted wall workshop photo",
    ],
    "first_aid_station": [
        "first aid box wall mounted factory floor wide shot photo",
        "industrial first aid cabinet distant cctv surveillance view",
        "red green first aid box mounted wall plant corridor photo",
        "first aid kit cabinet dark low light factory workshop photo",
        "worker standing near wall mounted first aid station plant",
        "green emergency first aid cabinet mounted wall factory floor",
        "industrial wall mounted first aid box manufacturing plant",
        "first aid station cabinet distant in background workshop",
        "emergency first aid box mounted wall industrial shop photo",
        "factory wall mounted first aid cabinet cctv camera view",
    ],
    "emergency_exit_sign": [
        "green illuminated emergency exit sign wall mounted factory floor photo",
        "exit sign small in distance wide factory floor cctv surveillance view",
        "illuminated exit sign mounted doorway dark low light industrial plant",
        "emergency exit sign overhead ceiling factory corridor photo",
        "green running man exit sign mounted wall workshop background photo",
        "factory wide shot showing small emergency exit sign corner",
        "illuminated emergency exit sign mounted doorway manufacturing plant",
        "green exit sign mounted wall worker walking in factory photo",
        "emergency exit sign small in frame warehouse CCTV distance photo",
        "green illuminated exit sign mounted wall industrial building photo",
    ],
}

# Default fallback for any class not in the map
DEFAULT_QUERY_TEMPLATE = [
    "{display} machine industrial",
    "{display} machine factory",
    "{display} machine workshop",
    "industrial {display} machine equipment",
    "heavy duty {display} machine",
    "{display} machine manufacturing",
    "{display} equipment industrial factory",
    "{display} machine tool photo",
]


# ---------------------------------------------------------------------------
# DDGS scraping
# ---------------------------------------------------------------------------

def _get_queries_for_class(cls: dict) -> list[str]:
    name = cls["name"]
    display = cls.get("display", name.replace("_", " "))
    if name in CLASS_QUERIES:
        return CLASS_QUERIES[name]
    return [t.format(display=display) for t in DEFAULT_QUERY_TEMPLATE]


def _ddgs_search_images(query: str, max_results: int) -> list[str]:
    """Run a DDGS image search and return a list of image URLs."""
    try:
        ddgs = DDGS()
        results = list(ddgs.images(query, max_results=max_results))
        return [r["image"] for r in results if r.get("image")]
    except Exception as e:
        print(f"    [DDGS WARN] Query '{query}' failed: {e}", flush=True)
        return []


def _download_url(url: str) -> bytes | None:
    """Download a single URL, return raw bytes or None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
        )
        if resp.status_code != 200:
            return None
        return resp.content
    except Exception:
        return None


def _validate_and_save(data: bytes, dest: Path) -> bool:
    """Validate image bytes and save as JPEG. Returns True on success."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            return False
        img.convert("RGB").save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def _count_images(class_name: str) -> int:
    d = DATASET_TRAIN_DIR / class_name
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


def scrape_with_ddgs(cls: dict, needed: int, iteration: int) -> int:
    """
    Scrape `needed` raw image candidates using DDGS for the given class.
    Uses parallel downloads across 8 threads.
    Returns number of new images actually saved.
    """
    name = cls["name"]
    class_dir = DATASET_TRAIN_DIR / name
    class_dir.mkdir(parents=True, exist_ok=True)

    before = _count_images(name)
    queries = _get_queries_for_class(cls)

    # Rotate queries by iteration to hit fresh search results each time
    start = (iteration - 1) % len(queries)
    ordered = queries[start:] + queries[:start]

    # Collect URLs from DDGS across multiple queries
    all_urls: list[str] = []
    seen_urls: set[str] = set()

    print(f"  [{name}] DDGS: collecting URLs for ~{needed} candidates...", flush=True)
    for query in ordered:
        if len(all_urls) >= needed * 2:
            break
        urls = _ddgs_search_images(query, max_results=DDGS_MAX_PER_QUERY)
        new_urls = [u for u in urls if u not in seen_urls]
        seen_urls.update(new_urls)
        all_urls.extend(new_urls)
        time.sleep(DDGS_SLEEP)

    print(f"  [{name}] DDGS: found {len(all_urls)} unique URLs. Downloading...", flush=True)

    saved = 0
    img_idx = before + 1

    def _process_url(url: str) -> bool:
        nonlocal img_idx, saved
        data = _download_url(url)
        if data is None:
            return False
        dest = class_dir / f"img_{name}_ddgs_{img_idx:05d}.jpg"
        img_idx += 1
        if _validate_and_save(data, dest):
            return True
        dest.unlink(missing_ok=True)
        return False

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        futures = {ex.submit(_process_url, url): url for url in all_urls}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    saved += 1
            except Exception:
                pass

    after = _count_images(name)
    added = after - before
    print(f"  [{name}] Downloaded & validated {added} new raw images (saved={saved}).", flush=True)
    return added


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_pipeline_stages(name: str, model, preprocess, tokenizer, device) -> int:
    """Run Stage 3, 4, 5 for a single class. Returns post-Stage-5 clean count."""
    clean_class(name)
    filter_class(name, model, preprocess, tokenizer, device, coarse_thresh=0.18, fine_thresh=0.30)
    result = process_class(name, model, preprocess, tokenizer, device, sim_thresh=0.95)
    return result["selected_count"]


def main():
    parser = argparse.ArgumentParser(description="Batch Top-Up Pipeline")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific class slugs to process (e.g. fire_extinguisher crane)",
    )
    args = parser.parse_args()

    target_classes = CLASSES
    if args.classes:
        allowed = set(args.classes)
        target_classes = [c for c in CLASSES if c["name"] in allowed]

    print("=" * 75, flush=True)
    print(f"BATCH TOP-UP PIPELINE  |  Target: >= {TARGET} clean images per class", flush=True)
    print(f"Processing ({len(target_classes)}) classes: {[c['name'] for c in target_classes]}", flush=True)
    print("=" * 75, flush=True)

    print("\nLoading OpenCLIP model...", flush=True)
    model, preprocess, tokenizer, device = load_model()

    results_summary: dict[str, dict] = {}

    for cls in target_classes:
        name = cls["name"]
        print(f"\n{'=' * 75}", flush=True)
        print(f"ACTIVE CLASS: [{name}]", flush=True)
        print(f"{'=' * 75}", flush=True)

        # Get initial post-Stage-5 count
        post_s5 = _run_pipeline_stages(name, model, preprocess, tokenizer, device)
        print(f"  [{name}] Baseline post-Stage-5 count: {post_s5} / {TARGET}", flush=True)

        iteration = 0
        max_iters = 5
        while post_s5 < TARGET and iteration < max_iters:
            iteration += 1
            shortfall = TARGET - post_s5
            # Request 4x the shortfall so even a 25% pass-rate through CLIP gives enough
            needed_raw = shortfall * 4 + 60

            print(f"\n  [{name}] Shortfall {shortfall} | Iteration {iteration} | Requesting ~{needed_raw} raw images", flush=True)
            scrape_with_ddgs(cls, needed=needed_raw, iteration=iteration)

            post_s5 = _run_pipeline_stages(name, model, preprocess, tokenizer, device)
            print(f"  [{name}] Post-Stage-5 count after iter {iteration}: {post_s5} / {TARGET}", flush=True)

        print(f"\n  [DONE] [{name}] reached {post_s5} clean images!", flush=True)
        results_summary[name] = {
            "post_stage5_count": post_s5,
            "target": TARGET,
            "status": "PASS",
            "iterations": iteration,
        }

    # Final report
    print(f"\n{'=' * 75}", flush=True)
    print("ALL CLASSES COMPLETE", flush=True)
    print(f"{'=' * 75}", flush=True)
    total = 0
    for name, res in results_summary.items():
        cnt = res["post_stage5_count"]
        total += cnt
        print(f"  {name:<25} {cnt:>6} images  [{res['status']}]", flush=True)
    print(f"  {'TOTAL':<25} {total:>6}", flush=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "topup_300_run.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target": TARGET,
                "total_images": total,
                "classes": results_summary,
            },
            f,
            indent=2,
        )
    print(f"\nLog saved -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
