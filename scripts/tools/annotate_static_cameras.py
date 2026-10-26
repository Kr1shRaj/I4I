"""
Auto-annotate fixed machines in fixed-camera CCTV frames (SAM 2.1 + worker subtraction)
=======================================================================================
For each camera in machine_prompts.yaml and each frame of its recordings:

1. SAM 2.1 segments every machine from its box prompt(s) (several prompts per machine are unioned,
   e.g. panel-saw body + sliding carriage), clipped to the machine's envelope box.
2. Pass 2 builds a per-camera "consensus" mask per machine (pixels present in >= 50% of frames).
   Static machines are clipped to the consensus dilated by --spill-px, so SAM can't spill onto stock.
3. Workers (COCO 'person' masks from a YOLO-seg model, dilated) and the burned-in OSD text are removed:
   labels cover VISIBLE machine pixels only (plan section 2).
4. An instance is written only if >= --min-visible of its consensus area is visible (plan: >= 40%).
5. Polygons: fragments < 10% of the instance dropped, each simplified, joined with zero-width bridges.

Output: <intake>/<source>/labels/<frame>.txt (YOLO-seg) and <intake>/annotation_report.csv.
Without --write it only renders review overlays.

These labels are machine-generated and spot-checked, not hand-drawn. Treat them as reviewed only
after a human QA pass, or accept the risk explicitly (see docs/factory_footage_analysis.md).

Usage
-----
    python scripts/tools/annotate_static_cameras.py --intake datasets/cctv_factory_intake --review-dir <dir>
    python scripts/tools/annotate_static_cameras.py --intake datasets/cctv_factory_intake --write
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel_cctv import mask_to_polygon  # noqa: E402

COLORS = {0: (0, 0, 255), 1: (255, 0, 255), 7: (0, 255, 0)}


def box_mask(shape, box):
    m = np.zeros(shape, bool)
    x1, y1, x2, y2 = box
    m[max(0, y1):y2, max(0, x1):x2] = True
    return m


def clean(m: np.ndarray, open_px: int = 5, close_px: int = 15) -> np.ndarray:
    """Remove SAM speckle: closing (fills pinholes), opening, then drop components < 2% of the mask."""
    m = m.astype(np.uint8)
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    total = stats[1:, cv2.CC_STAT_AREA].sum()
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.02 * total]
    return np.isin(lab, keep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intake", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=None, help="default <intake>/machine_prompts.yaml")
    ap.add_argument("--sam", type=Path, default=PROJECT_ROOT / "models" / "pretrained" / "sam2.1_l.pt")
    ap.add_argument("--person-model", type=Path, default=PROJECT_ROOT / "models" / "pretrained" / "yolo26s-seg.pt")
    ap.add_argument("--person-conf", type=float, default=0.25)
    ap.add_argument("--person-dilate", type=int, default=9)
    ap.add_argument("--min-visible", type=float, default=0.40)
    ap.add_argument("--spill-px", type=int, default=15)
    ap.add_argument("--simplify", type=float, default=0.002)
    ap.add_argument("--static-clip", nargs="*", default=["hydraulic_press", "edge_banding_machine", "panel_saw"],
                    help="machines clipped to their dilated consensus mask")
    ap.add_argument("--write", action="store_true", help="write labels/ (otherwise review overlays only)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing label files")
    ap.add_argument("--review-dir", type=Path, default=None, help="save overlays here (every frame)")
    args = ap.parse_args()

    from ultralytics import SAM, YOLO
    cfg = yaml.safe_load((args.prompts or args.intake / "machine_prompts.yaml").read_text(encoding="utf-8"))
    class_ids = cfg["classes"]
    sam, people = SAM(str(args.sam)), YOLO(str(args.person_model))
    if args.review_dir:
        args.review_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for cam, ccfg in cfg["cameras"].items():
        frames = [p for s in ccfg["sources"] for p in sorted((args.intake / s / "images").glob("*.jpg"))]
        machines = list(ccfg["machines"])
        print(f"{cam}: {len(frames)} frames, machines {machines}", flush=True)

        # Pass 1: raw SAM masks (packed bits) and worker masks
        raw = {m: [] for m in machines}
        person_masks, shape = [], None
        for i, f in enumerate(frames):
            im = cv2.imread(str(f)); shape = im.shape[:2]
            boxes, owners = [], []
            for m in machines:
                for b in ccfg["machines"][m]["prompts"]:
                    boxes.append(b); owners.append(m)
            r = sam(im, bboxes=boxes, verbose=False)[0]
            masks = r.masks.data.cpu().numpy() > 0.5
            for m in machines:
                env = box_mask(shape, ccfg["machines"][m]["envelope"])
                u = np.zeros(shape, bool)
                for mk, o in zip(masks, owners):
                    if o == m:
                        u |= mk
                raw[m].append(np.packbits(clean(u & env)))
            pr = people.predict(im, classes=[0], conf=args.person_conf, imgsz=1280, retina_masks=True, verbose=False)[0]
            pm = np.zeros(shape, bool)
            if pr.masks is not None:
                for mk in pr.masks.data.cpu().numpy():
                    pm |= cv2.resize(mk, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            if args.person_dilate:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.person_dilate, args.person_dilate))
                pm = cv2.dilate(pm.astype(np.uint8), k).astype(bool)
            person_masks.append(np.packbits(pm))
            if (i + 1) % 50 == 0:
                print(f"  pass 1: {i + 1}/{len(frames)}", flush=True)

        unpack = lambda b: np.unpackbits(b)[: shape[0] * shape[1]].reshape(shape).astype(bool)
        osd = np.zeros(shape, bool)
        for b in ccfg.get("osd_boxes", []):
            osd |= box_mask(shape, b)

        # Pass 2: consensus per machine
        consensus = {}
        for m in machines:
            acc = np.zeros(shape, np.uint16)
            for b in raw[m]:
                acc += unpack(b)
            consensus[m] = acc >= 0.5 * len(frames)
            print(f"  consensus {m}: {int(consensus[m].sum())} px", flush=True)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.spill_px + 1, 2 * args.spill_px + 1))
        allowed = {m: cv2.dilate(consensus[m].astype(np.uint8), k).astype(bool) if m in args.static_clip
                   else np.ones(shape, bool) for m in machines}

        # Pass 3: finalise, write
        for i, f in enumerate(frames):
            pm = unpack(person_masks[i])
            lines, row = [], {"frame": f.name, "source": f.parent.parent.name, "camera": cam}
            vis_img = cv2.imread(str(f)) if args.review_dir else None
            for m in machines:
                mk = unpack(raw[m][i]) & allowed[m] & ~pm & ~osd
                mk = clean(mk)
                c_area = max(1, consensus[m].sum())
                visible = (mk & consensus[m]).sum() / c_area
                hidden_by_workers = (consensus[m] & pm).sum() / c_area
                row[f"{m}_visible"] = round(float(visible), 3)
                row[f"{m}_hidden_by_workers"] = round(float(hidden_by_workers), 3)
                status = "labelled"
                if visible < args.min_visible and hidden_by_workers < 1 - args.min_visible:
                    # SAM missed a machine that workers don't hide: fall back to the static outline
                    mk = clean(consensus[m] & ~pm & ~osd)
                    visible = (mk & consensus[m]).sum() / c_area
                    status = "labelled (static fallback)"
                if visible < args.min_visible:
                    row[f"{m}_status"] = "dropped (<min visible)"
                    continue
                poly = mask_to_polygon(mk, args.simplify)
                if poly is None or len(poly) < 3:
                    row[f"{m}_status"] = "dropped (no polygon)"
                    continue
                cid = class_ids[m]
                norm = np.clip(poly.astype(np.float64) / [shape[1], shape[0]], 0, 1)
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in norm.flatten()))
                row[f"{m}_status"] = status
                if vis_img is not None:
                    vis_img[mk] = (0.55 * vis_img[mk] + 0.45 * np.array(COLORS[cid])).astype(np.uint8)
                    cv2.polylines(vis_img, [poly.astype(np.int32)], True, COLORS[cid], 2)
            row["instances"] = len(lines)
            row["workers_px"] = int(pm.sum())
            report.append(row)
            if args.write:
                lbl_dir = f.parent.parent / "labels"
                lbl_dir.mkdir(exist_ok=True)
                lp = lbl_dir / f"{f.stem}.txt"
                if lp.exists() and not args.overwrite:
                    raise SystemExit(f"{lp} exists; pass --overwrite to replace reviewed labels")
                lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if vis_img is not None:
                cv2.imwrite(str(args.review_dir / f"{f.stem}.jpg"), cv2.resize(vis_img, (960, 540)),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])

    fields = sorted({k for r in report for k in r}, key=lambda k: (k not in ("frame", "source", "camera"), k))
    with (args.intake / "annotation_report.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(report)
    for cam in cfg["cameras"]:
        rows = [r for r in report if r["camera"] == cam]
        for m in cfg["cameras"][cam]["machines"]:
            n = sum(1 for r in rows if str(r.get(f"{m}_status", "")).startswith("labelled"))
            fb = sum(1 for r in rows if r.get(f"{m}_status") == "labelled (static fallback)")
            print(f"{cam} {m}: labelled in {n}/{len(rows)} frames ({fb} via static fallback)")
    print(f"{'Labels written' if args.write else 'Dry run (no labels written)'}; report: {args.intake / 'annotation_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
