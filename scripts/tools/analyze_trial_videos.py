"""
Trial Video Diagnostics: Machine Mask Quality + Fire Extinguisher Detection Disparity
====================================================================================
Inputs (screen recordings of the CCTV annotator, overlays baked in):
  trial_videos/Screen Recording 2026-09-28 124839.mp4   workshop machine (panel saw)
  trial_videos/Screen Recording 2026-09-28 125921.mp4   handheld / close-up fire extinguisher
  feature_maps_9class/inputs/cctv_screenshot_2026-09-30.png   CCTV room (missed extinguisher)

Outputs -> trial_videos/analysis/
  machine_fXXXX_mask_diagnosis.png   drawn outline vs model mask, raw mask probability, box crop, polygonisation
  machine_mask_summary.json
  panel_saw_training_labels.jpg      contact sheet of training polygons for the detected class
  extinguisher_comparison.png        trial vs CCTV extinguisher crops + measured metrics
  extinguisher_degradation_sweep.png confidence vs object height / blur / colour
  extinguisher_summary.json
  smooth_mask_demo.png               model mask polygon: default vs smoothed

The annotator's purple/yellow depth overlay is baked into every recorded frame. The model never saw it live,
so frames are de-tinted to grayscale before inference unless noted. This is a proxy, not the live input.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt"
VIDEOS = PROJECT_ROOT.parent / "trial_videos"
OUT = VIDEOS / "analysis"
VIDEO_MACHINE = VIDEOS / "Screen Recording 2026-09-28 124839.mp4"
VIDEO_FE = VIDEOS / "Screen Recording 2026-09-28 125921.mp4"
CCTV_SHOT = PROJECT_ROOT / "archive" / "analysis_outputs" / "feature_maps_9class" / "inputs" / "cctv_screenshot_2026-09-30.png"
TRAIN_LBL = PROJECT_ROOT / "datasets" / "9class" / "train_9class" / "labels"
TRAIN_IMG = PROJECT_ROOT / "datasets" / "9class" / "train_9class" / "images"

CLASS_NAMES = ["panel_saw", "hydraulic_press", "table_saw", "planer", "jointer",
               "band_saw", "miter_saw", "edge_banding_machine", "fire_extinguisher"]
IMGSZ = 960
MACHINE_FRAMES = [186, 1210, 1303]   # 558 is identical to 186 (paused stream)
MACHINE_VIEW = (7, 180, 1744, 795)         # x0, y0, x1, y1 of the camera image inside the 1918x916 recording
STREAM_SIZE = (480, 360)                   # stream resolution reported by the annotator status bar
FE_FRAMES = [518, 561, 604, 648]
# Extinguisher boxes in the 1907x911 CCTV screenshot (x0, y0, x1, y1), located by colour blob + visual check.
# cam03: upright, hanging by the wall. cam02: lying tilted on the floor, cut off by the bottom edge of the tile.
CCTV_FE = {"cam03": (445, 722, 466, 763), "cam02": (1565, 438, 1622, 497)}
CCTV_TILE_H = 393                          # camera tile height in the screenshot (px)
CCTV_TILE_W = 866


def gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {path.name}")
    return frame


def drawn_outline_mask(view):
    """Recover the annotator's drawn polygon: thin green-cyan strokes -> filled region."""
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    stroke = ((hsv[..., 0] >= 60) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 180) & (hsv[..., 2] > 100)).astype(np.uint8) * 255
    lines = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros(view.shape[:2], np.uint8)
    cv2.drawContours(filled, [c], -1, 1, thickness=-1)
    approx = cv2.approxPolyDP(c, 2.0, True)
    return filled, approx


class RawSeg:
    """Direct forward pass to read boxes, mask coefficients and prototypes before any cropping/thresholding."""
    def __init__(self, weights, device):
        self.net = YOLO(str(weights)).model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img):
        h, w = img.shape[:2]
        s = IMGSZ / max(h, w)
        nh, nw = round(h * s / 32) * 32, round(w * s / 32) * 32
        inp = cv2.resize(img, (nw, nh))
        x = torch.from_numpy(inp[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(self.device)
        (dets, proto), _ = self.net(x)
        return inp, dets[0].cpu().numpy(), proto[0].cpu().numpy()


def top_det(dets, cls=None, min_conf=0.05):
    d = dets[dets[:, 4] >= min_conf]
    if cls is not None:
        d = d[d[:, 5].astype(int) == cls]
    return d[np.argmax(d[:, 4])] if len(d) else None


def mask_prob(det, proto, shape):
    logit = np.tensordot(det[6:6 + proto.shape[0]], proto, axes=1)
    up = cv2.resize(logit, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return 1 / (1 + np.exp(-up)), logit


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


# ============================================================================ machine masks
def analyse_machine(model, raw, device):
    rows = []
    for f in MACHINE_FRAMES:
        frame = read_frame(VIDEO_MACHINE, f)
        x0, y0, x1, y1 = MACHINE_VIEW
        view = frame[y0:y1, x0:x1]
        drawn, drawn_poly = drawn_outline_mask(view)
        cv2.imwrite(str(OUT / f"machine_f{f:04d}_recorded.jpg"), view)

        # The UI stretches the 480x360 stream to 2.83:1. Only the tinted frame restored to 4:3 reproduces the live
        # panel_saw detection (0.56-0.63 vs 0.79 live); grayscale or stretched inputs lose it entirely.
        variants = {"tinted, restored to 480x360": cv2.resize(view, STREAM_SIZE, interpolation=cv2.INTER_AREA)}
        best = None
        for vname, img in variants.items():
            inp, dets, proto = raw(img)
            det = top_det(dets, cls=0)
            if det is None:
                continue
            prob, logit = mask_prob(det, proto, inp.shape)
            box = det[:4].astype(int)
            in_box = np.zeros(prob.shape, bool); in_box[box[1]:box[3], box[0]:box[2]] = True
            model_mask = (prob > 0.5) & in_box
            disp = lambda m: cv2.resize(m.astype(np.uint8), (view.shape[1], view.shape[0]), interpolation=cv2.INTER_NEAREST)
            score = iou(disp(model_mask), drawn) if drawn is not None else 0.0
            cand = dict(variant=vname, img=img, inp=inp, det=det, prob=prob, logit=logit, proto=proto,
                        box=box, model_mask=model_mask, iou_vs_drawn=score,
                        cut_by_box=float(((prob > 0.5) & ~in_box).sum() / max(1, (prob > 0.5).sum())))
            if best is None or score > best["iou_vs_drawn"]:
                best = cand
        if best is None:
            print(f"frame {f}: no detection")
            continue

        # Ultralytics polygonisation of the same detection (what masks.xy returns)
        res = model.predict(best["img"], imgsz=IMGSZ, conf=0.05, retina_masks=False, verbose=False)[0]
        res_r = model.predict(best["img"], imgsz=IMGSZ, conf=0.05, retina_masks=True, verbose=False)[0]
        pick = lambda r: int(np.argmax(np.where(r.boxes.cls.cpu().numpy() == 0, r.boxes.conf.cpu().numpy(), -1)))
        k = pick(res)
        mk = res.masks.data[k].cpu().numpy()
        n_pieces = cv2.connectedComponents(mk.astype(np.uint8))[0] - 1
        xy_all = res.masks.xy[k]
        # masks.xy itself is scaled to the original image, so convert the input-resolution segment the same way
        xy_largest = ops.scale_coords(mk.shape, ops.masks2segments(res.masks.data[k:k + 1], strategy="largest")[0],
                                      res.orig_shape, normalize=False)
        mk_r = res_r.masks.data[pick(res_r)].cpu().numpy()

        # vertical profile of mask probability inside the box (does confidence decay towards the base?)
        bx = best["box"]
        prof = best["prob"][bx[1]:bx[3], bx[0]:bx[2]].mean(axis=1)
        thirds = [float(np.mean(t)) for t in np.array_split(prof, 3)]

        row = {"frame": f, "variant": best["variant"], "class": CLASS_NAMES[int(best["det"][5])],
               "conf": round(float(best["det"][4]), 3),
               "box_xyxy_on_input": bx.tolist(), "input_size": list(best["inp"].shape[1::-1]),
               "mask_area_above_0.5_outside_box_pct": round(100 * best["cut_by_box"], 1),
               "mask_prob_mean_by_box_third_top_mid_bottom": [round(t, 3) for t in thirds],
               "mask_pieces": int(n_pieces),
               "masks_xy_vertices_strategy_all": int(len(xy_all)),
               "masks_xy_vertices_strategy_largest": int(len(xy_largest)),
               "mask_res_retina_false": list(mk.shape[::-1]), "mask_res_retina_true": list(mk_r.shape[::-1]),
               "proto_res": list(best["proto"].shape[1:][::-1])}
        row.update(simplification_fit(view, best))
        rows.append(row)
        render_machine(f, view, drawn, best, res, k, xy_all, xy_largest, prof, row)
        if f == MACHINE_FRAMES[0]:
            smooth_demo(best, res, k)
        print(f"frame {f}: {row}")
    (OUT / "machine_mask_summary.json").write_text(json.dumps(rows, indent=2))
    return rows


def stroke_points(view):
    """Pixels of the annotator's green outline (sampled; the line breaks up where it crosses the green tint)."""
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    s = (hsv[..., 0] >= 60) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 180) & (hsv[..., 2] > 100)
    ys, xs = np.nonzero(s)
    return np.stack([xs, ys], 1).astype(np.float32)


def simplification_fit(view, best):
    """Is the drawn outline the model's mask, simplified? Measure the distance (in 480x360 stream pixels) from the
    drawn-line pixels to the model's mask contour, simplified with increasing approxPolyDP tolerance."""
    pts = stroke_points(view)
    pts = pts * np.array([STREAM_SIZE[0] / view.shape[1], STREAM_SIZE[1] / view.shape[0]], np.float32)
    inp = best["inp"]
    m = cv2.resize(best["model_mask"].astype(np.uint8), STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    fit = []
    for eps in (0, 1, 2, 3, 4, 6, 8):
        poly = c if eps == 0 else cv2.approxPolyDP(c, eps, True)
        d = [abs(cv2.pointPolygonTest(poly, (float(x), float(y)), True)) for x, y in pts[::3]]
        fit.append({"tolerance_px": eps, "corners": int(len(poly)), "median_dist_px": round(float(np.median(d)), 2),
                    "p90_dist_px": round(float(np.percentile(d, 90)), 2)})
    return {"drawn_line_px_sampled": int(len(pts)), "simplification_fit": fit,
            "model_contour_points_at_stream_res": int(len(c))}


def render_machine(f, view, drawn, best, res, k, xy_all, xy_largest, prof, row):
    inp = best["inp"]
    fig, ax = plt.subplots(2, 3, figsize=(21, 10), dpi=100)
    ax = ax.ravel()
    v = view.copy()
    for x, y in stroke_points(view).astype(int):
        cv2.circle(v, (int(x), int(y)), 2, (0, 255, 255), -1)
    ax[0].imshow(v[:, :, ::-1]); ax[0].set_title(f"1. Recorded frame {f}: annotator's outline pixels (yellow)")
    ax[1].imshow(inp[:, :, ::-1]); ax[1].imshow(best["prob"], cmap="jet", alpha=0.5, vmin=0, vmax=1)
    x1, y1, x2, y2 = best["box"]
    ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=2, ls="--"))
    ax[1].set_title(f"2. Raw mask probability, no crop/threshold ({row['class']} {row['conf']:.2f})\nwhite = predicted box (mask is cut to it)")
    m = inp.copy(); m[best["model_mask"]] = (0.4 * m[best["model_mask"]] + 0.6 * np.array([60, 220, 60])).astype(np.uint8)
    ax[2].imshow(m[:, :, ::-1]); ax[2].set_title(f"3. Final mask (p>0.5, cropped to box): {row['mask_area_above_0.5_outside_box_pct']}% of confident mask cut off")
    ax[3].imshow(best["logit"], cmap="coolwarm", vmin=-abs(best["logit"]).max(), vmax=abs(best["logit"]).max(), interpolation="nearest")
    ax[3].set_title(f"4. Mask logit at prototype resolution {row['proto_res'][0]}x{row['proto_res'][1]} (nearest)")
    ax[4].plot(prof, np.arange(len(prof))); ax[4].invert_yaxis(); ax[4].axvline(0.5, ls="--", c="k")
    ax[4].set_xlim(0, 1); ax[4].set_xlabel("mean mask probability across the box width"); ax[4].set_ylabel("row inside box (top -> bottom)")
    ax[4].set_title("5. Does mask confidence fade towards the base?")
    p = inp.copy()
    s = np.array([inp.shape[1] / res.orig_shape[1], inp.shape[0] / res.orig_shape[0]])
    cv2.polylines(p, [(xy_all * s).astype(np.int32)], True, (0, 0, 255), 1)
    cv2.polylines(p, [(xy_largest * s).astype(np.int32)], True, (0, 255, 0), 1)
    ax[5].imshow(p[:, :, ::-1])
    ax[5].set_title(f"6. masks.xy polygon: red = default 'all' ({row['masks_xy_vertices_strategy_all']} pts, {row['mask_pieces']} pieces joined)\ngreen = 'largest' piece only")
    for a in ax[[0, 1, 2, 3, 5]]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"machine_f{f:04d}_mask_diagnosis.png")
    plt.close(fig)


def smooth_mask(det, proto, out_hw, box, blur_sigma=1.5, min_piece_frac=0.02):
    """Sub-pixel smooth mask: bicubic-upsample the logit, Gaussian-smooth, threshold at 0, crop to box,
    drop tiny fragments, then trace with full-resolution contours (no vertex decimation)."""
    logit = np.tensordot(det[6:6 + proto.shape[0]], proto, axes=1).astype(np.float32)
    up = cv2.resize(logit, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), blur_sigma)
    m = (up > 0).astype(np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    crop = np.zeros_like(m); crop[max(0, y1):y2, max(0, x1):x2] = 1
    m &= crop
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_piece_frac * m.sum():
            keep[lab == i] = 1
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return keep, cnts


def smooth_demo(best, res, k):
    inp = best["inp"]
    keep, cnts = smooth_mask(best["det"], best["proto"], inp.shape[:2], best["box"])
    a = inp.copy(); b = inp.copy()
    s = np.array([inp.shape[1] / res.orig_shape[1], inp.shape[0] / res.orig_shape[0]])
    cv2.polylines(a, [(res.masks.xy[k] * s).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.drawContours(b, cnts, -1, (0, 220, 0), 2)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5), dpi=100)
    ax[0].imshow(a[:, :, ::-1]); ax[0].set_title(f"Default: masks.xy ('all' strategy joins {len(cnts)}+ pieces with bridges)")
    ax[1].imshow(b[:, :, ::-1]); ax[1].set_title(f"Smoothed: cubic logit upsample + Gaussian + per-piece contours ({len(cnts)} pieces)")
    for x in ax:
        x.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "smooth_mask_demo.png"); plt.close(fig)


def training_label_sheet(cls=0, n=24):
    files = sorted(p for p in TRAIN_LBL.glob("*.txt")
                   if p.read_text().strip() and all(int(l.split()[0]) == cls for l in p.read_text().splitlines() if l.strip()))
    rng = np.random.default_rng(1)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    tiles = []
    for i in pick:
        lp = files[i]
        im = cv2.imread(str(TRAIN_IMG / (lp.stem + ".jpg")))
        if im is None:
            continue
        h, w = im.shape[:2]
        ov = im.copy()
        for line in lp.read_text().splitlines():
            pts = (np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(ov, [pts], (60, 220, 60))
            cv2.polylines(im, [pts], True, (0, 160, 0), max(2, w // 300))
        im = cv2.addWeighted(im, 0.55, ov, 0.45, 0)
        tiles.append(cv2.resize(im, (320, 240)))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])
    cv2.imwrite(str(OUT / f"{CLASS_NAMES[cls]}_training_labels.jpg"), sheet)


# ============================================================================ extinguisher
def object_metrics(crop_bgr, obj_mask):
    """Scale-aware appearance metrics for an object region inside a crop."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.where(obj_mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    ring = cv2.dilate(obj_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~obj_mask
    contrast = abs(g[obj_mask].mean() - g[ring].mean()) / (g[obj_mask].mean() + g[ring].mean() + 1e-6)
    # sharpness at a common scale: resize object box to 128 px tall, variance of Laplacian
    box = crop_bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    norm = cv2.resize(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), (max(8, int(128 * w / h)), 128), interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    # edge width: intensity step / steepest gradient on the boundary, as % of object height
    edge = cv2.morphologyEx(obj_mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = np.hypot(gx, gy)
    step = abs(g[obj_mask].mean() - g[ring].mean())
    edge_w = step / (np.percentile(grad[edge], 90) + 1e-6)
    hue = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return {"height_px": int(h), "width_px": int(w), "aspect_h_over_w": round(h / w, 2),
            "contrast_vs_surround": round(float(contrast), 3),
            "sharpness_laplacian_var_at_128px": round(lap, 1),
            "edge_width_px": round(float(edge_w), 2), "edge_width_pct_of_height": round(float(100 * edge_w / h), 2),
            "mean_saturation": round(float(hue[..., 1][obj_mask].mean()), 1)}


def red_object_mask(crop, min_area=30):
    """Extinguisher body: reddish / magenta (tint-shifted) pixels, largest blob."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((hch < 12) | (hch > 150)) & (s > 70) & (v > 50)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i) if st[i, cv2.CC_STAT_AREA] >= min_area else None


def analyse_extinguisher(model):
    summary = {}
    # trial video
    trial = []
    for f in FE_FRAMES:
        fr = read_frame(VIDEO_FE, f)
        for src, img in (("recorded (tinted)", fr), ("grayscale", gray3(fr))):
            r = model.predict(img, imgsz=IMGSZ, conf=0.05, verbose=False)[0]
            confs = [float(c) for k, c in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()) if k == 8]
            trial.append({"frame": f, "input": src, "fire_extinguisher_conf": round(max(confs), 3) if confs else 0.0})
    summary["trial_offline_redetection"] = trial
    fr = read_frame(VIDEO_FE, FE_FRAMES[0])
    m_trial = red_object_mask(fr)
    met_trial = object_metrics(fr, m_trial)
    met_trial["object_area_pct_of_visible_view"] = round(100 * m_trial.mean(), 2)
    met_trial["view_size"] = list(fr.shape[1::-1])
    summary["trial_video"] = met_trial

    shot = cv2.imread(str(CCTV_SHOT))
    cctv_crops = {}
    pad = 12
    for cam, (x0, y0, x1, y1) in CCTV_FE.items():
        crop = shot[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        inside = np.zeros(crop.shape[:2], bool); inside[pad:-pad, pad:-pad] = True
        # extinguisher = darker, more saturated magenta than the tinted floor/wall around it
        m = inside & (hsv[..., 1] > np.percentile(hsv[..., 1][~inside], 75)) & (hsv[..., 2] < np.percentile(hsv[..., 2][~inside], 40))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        met = object_metrics(crop, m)
        met["box_in_screenshot_xyxy"] = [x0, y0, x1, y1]
        met["box_w_h_screenshot_px"] = [x1 - x0, y1 - y0]
        met["height_in_480_line_stream_px"] = round((y1 - y0) * 480 / CCTV_TILE_H, 1)
        met["object_area_pct_of_camera_frame"] = round(100 * (x1 - x0) * (y1 - y0) / (CCTV_TILE_W * CCTV_TILE_H), 3)
        summary[cam] = met
        cctv_crops[cam] = (crop, m)

    # degradation sweep: shrink each trial frame so the extinguisher is h px tall inside an 854x480 frame,
    # JPEG-compress like a CCTV substream, and average the confidence over the 4 frames
    sweep = []
    H0 = met_trial["height_px"]
    frames = [read_frame(VIDEO_FE, f) for f in FE_FRAMES]
    for colour in ("colour (tinted)", "grayscale"):
        for blur in (0.0, 1.2):
            for h_target in (240, 160, 120, 80, 60, 40, 30, 20):
                confs_f = []
                for fr_k in frames:
                    base = fr_k if colour.startswith("colour") else gray3(fr_k)
                    s = h_target / H0
                    small = cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    if blur:
                        small = cv2.GaussianBlur(small, (0, 0), blur)
                    canvas = np.full((480, 854, 3), base.reshape(-1, 3).mean(0), np.uint8)
                    ch, cw = min(480, small.shape[0]), min(854, small.shape[1])
                    oy, ox = (480 - ch) // 2, (854 - cw) // 2
                    canvas[oy:oy + ch, ox:ox + cw] = small[:ch, :cw]
                    ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 45])
                    canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                    r = model.predict(canvas, imgsz=IMGSZ, conf=0.01, verbose=False)[0]
                    c8 = [float(c) for k, c in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()) if k == 8]
                    confs_f.append(max(c8) if c8 else 0.0)
                    if colour.startswith("colour") and blur and h_target in (120, 40) and fr_k is frames[-1]:
                        cv2.imwrite(str(OUT / f"sweep_example_h{h_target}.jpg"), canvas)
                sweep.append({"colour": colour, "blur_sigma": blur, "height_px": h_target,
                              "conf": round(float(np.mean(confs_f)), 3), "conf_per_frame": [round(c, 3) for c in confs_f]})
    summary["degradation_sweep"] = sweep
    (OUT / "extinguisher_summary.json").write_text(json.dumps(summary, indent=2))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2), dpi=100)
    vis = fr.copy(); cnts, _ = cv2.findContours(m_trial.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    ax[0].imshow(vis[:, :, ::-1]); ax[0].set_title(f"Trial video (detected live 76-86%)\nbody {met_trial['height_px']} px tall in a {fr.shape[1]}x{fr.shape[0]} crop of the view")
    for j, cam in enumerate(("cam03", "cam02")):
        if cam in cctv_crops:
            crop, m = cctv_crops[cam]
            big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            cn, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(big, cn, -1, (0, 255, 255), 1)
            met = summary[cam]
            ax[1 + j].imshow(big[:, :, ::-1])
            ax[1 + j].set_title(f"CCTV {cam} (missed), shown 4x\nbody ~{met['height_in_480_line_stream_px']} px tall in the 480-line stream")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "extinguisher_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=110)
    for colour, ls in (("colour (tinted)", "-"), ("grayscale", "--")):
        for blur, mk in ((0.0, "o"), (1.2, "s")):
            pts = [(d["height_px"], d["conf"]) for d in sweep if d["colour"] == colour and d["blur_sigma"] == blur]
            ax.plot(*zip(*pts), ls=ls, marker=mk, label=f"{colour}, blur sigma {blur}")
    for cam in ("cam03", "cam02"):
        if "height_in_480_line_stream_px" in summary.get(cam, {}):
            ax.axvline(summary[cam]["height_in_480_line_stream_px"], c="r", alpha=.4)
            ax.text(summary[cam]["height_in_480_line_stream_px"], 0.95, f" CCTV {cam}", color="r", fontsize=8, va="top")
    ax.axhline(0.25, c="k", ls=":", lw=1); ax.text(235, 0.27, "conf 0.25", fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_xticks([20, 30, 40, 60, 80, 120, 160, 240]); ax.set_xticklabels([20, 30, 40, 60, 80, 120, 160, 240])
    ax.set_xlabel("extinguisher height in a 480-line frame (px)"); ax.set_ylabel("fire_extinguisher confidence")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title("Trial-video extinguisher shrunk to CCTV sizes (JPEG q45, 854x480 frame)")
    fig.tight_layout(); fig.savefig(OUT / "extinguisher_degradation_sweep.png"); plt.close(fig)
    return summary


# Scenes whose live detections could not be reproduced offline (heavy overlay): analyse the drawn outline only.
OUTLINE_ONLY = {838: ("panel_saw 58% on a long cabinet machine", (7, 180, 1744, 795)),
                980: ("hydraulic_press 74% on a press", (7, 77, 1744, 896))}


def analyse_outlines():
    """Save the scenes whose live detections could not be reproduced offline, side by side, for visual review."""
    fig, ax = plt.subplots(1, len(OUTLINE_ONLY), figsize=(11 * len(OUTLINE_ONLY), 5.5), dpi=100)
    for a, (f, (desc, (x0, y0, x1, y1))) in zip(np.atleast_1d(ax), OUTLINE_ONLY.items()):
        view = read_frame(VIDEO_MACHINE, f)[y0:y1, x0:x1]
        cv2.imwrite(str(OUT / f"machine_f{f:04d}_recorded.jpg"), view)
        a.imshow(view[:, :, ::-1]); a.axis("off")
        a.set_title(f"frame {f}: live {desc} (not reproducible offline)")
    fig.tight_layout(); fig.savefig(OUT / "unreproduced_scenes.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))
    raw = RawSeg(WEIGHTS, device)
    analyse_machine(model, raw, device)
    analyse_outlines()
    training_label_sheet(0)
    s = analyse_extinguisher(model)
    print(json.dumps({k: v for k, v in s.items() if k != "degradation_sweep"}, indent=2))
    print("sweep:", [(d["colour"][:4], d["blur_sigma"], d["height_px"], d["conf"]) for d in s["degradation_sweep"]])


if __name__ == "__main__":
    main()
