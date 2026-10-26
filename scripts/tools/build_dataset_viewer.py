"""
scripts/tools/build_dataset_viewer.py
=====================================
Build a browser gallery of the active 9-class dataset (masks, boxes, filters, zoom).

1. reads the lists that define the dataset (never the image folders, see docs/dataset_layout_9class.md):
     train_oversampled_9class.txt  -> deduplicated; entries are tagged train / highangle / hard_negative
     valid_9class.txt, test_9class.txt
     cctv_benchmark.txt            (frozen factory CCTV benchmark, evaluation only)
2. parses every label file (YOLO-seg polygons, or plain boxes) into class, box, polygon and box area %
3. writes docs/viewer/dataset_manifest.json and docs/viewer/index.html
4. checks the counts against datasets/9class/build_9class_stats.json and fails on missing labels

The page loads the manifest and images over HTTP, so serve the project root:

    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\build_dataset_viewer.py
    python -m http.server 8000 --directory .
    -> http://localhost:8000/docs/viewer/index.html
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
OUT_DIR = PROJECT_ROOT / "docs" / "viewer"

# split tag -> list file; train is subdivided by where the image lives
LISTS = {
    "train": DATASET_DIR / "train_oversampled_9class.txt",
    "val": DATASET_DIR / "valid_9class.txt",
    "test": DATASET_DIR / "test_9class.txt",
    "cctv": DATASET_DIR / "cctv_benchmark.txt",
}
SPLIT_ORDER = ["train", "highangle", "hard_negative", "val", "test", "cctv"]
THUMB_EPS_PX = 1.0  # thumbnail polygon simplification tolerance (image pixels)


def list_paths(list_file: Path) -> list[Path]:
    """Same resolution as Ultralytics: a leading ./ is the list file's folder (string join, junctions untouched)."""
    out = []
    for line in list_file.read_text().split():
        p = line[2:] if line.startswith("./") else line
        out.append(Path(os.path.normpath(list_file.parent / p)))
    return out


def label_path(img: Path) -> Path:
    s = str(img)
    i = s.rfind(f"{os.sep}images{os.sep}")
    return Path(s[:i] + f"{os.sep}labels{os.sep}" + s[i + 8:]).with_suffix(".txt")


def train_tag(img: Path) -> str:
    parts = img.parts
    if "highangle" in parts:
        return "highangle"
    if "hard_negatives" in parts:
        return "hard_negative"
    return "train"


def simplify(poly: list[float], w: int, h: int, eps_px: float = THUMB_EPS_PX) -> list[float]:
    """Douglas-Peucker in image pixels, for drawing thumbnails. The modal draws the exact label file."""
    pts = (np.array(poly).reshape(-1, 2) * [w, h]).astype(np.float32).reshape(-1, 1, 2)
    s = cv2.approxPolyDP(pts, eps_px, True).reshape(-1, 2)
    if len(s) < 3:
        return poly
    return [round(float(x), 4) for x in (s / [w, h]).ravel()]


def parse_label(lp: Path, w: int, h: int) -> list[dict]:
    objs = []
    for line in lp.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        c, xs = int(v[0]), [float(x) for x in v[1:]]
        if len(xs) == 4:  # plain box: cx cy w h
            cx, cy, w, h = xs
            poly = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
        else:
            poly = xs
        px, py = poly[0::2], poly[1::2]
        x1, y1, x2, y2 = min(px), min(py), max(px), max(py)
        objs.append({
            "c": c,
            "b": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            "a": round((x2 - x1) * (y2 - y1) * 100, 2),
            "v": len(poly) // 2,          # vertex count in the label file
            "p": simplify(poly, w, h),    # thumbnail polygon
        })
    return objs


def main() -> int:
    names = yaml.safe_load((DATASET_DIR / "data_9class.yaml").read_text())["names"]
    names = [names[i] for i in range(len(names))]

    entries: dict[Path, str] = {}
    oversample = Counter()
    for split, lf in LISTS.items():
        for img in list_paths(lf):
            tag = train_tag(img) if split == "train" else split
            if split == "train":
                oversample[img] += 1
            if img in entries and entries[img] != tag:
                print(f"WARNING: {img.name} is in both {entries[img]} and {tag}")
            entries.setdefault(img, tag)

    images, missing_img, missing_lbl, bad_cls = [], [], [], []
    for img, tag in entries.items():
        lp = label_path(img)
        if not img.exists():
            missing_img.append(str(img))
            continue
        if not lp.exists():
            missing_lbl.append(str(lp))
            continue
        with Image.open(img) as im:
            w, h = im.size
        objs = parse_label(lp, w, h)
        bad_cls += [str(lp) for o in objs if not 0 <= o["c"] < len(names)]
        images.append({
            "f": Path(os.path.relpath(img, OUT_DIR)).as_posix(),  # served paths relative to index.html
            "l": Path(os.path.relpath(lp, OUT_DIR)).as_posix(),
            "s": tag,
            "w": w,
            "h": h,
            "n": oversample.get(img, 1),
            "o": objs,
        })
    images.sort(key=lambda r: (SPLIT_ORDER.index(r["s"]), r["f"].rsplit("/", 1)[-1]))

    # ---- summary ----
    per_split = {}
    for s in SPLIT_ORDER:
        rows = [r for r in images if r["s"] == s]
        per_split[s] = {
            "images": len(rows),
            "positive": sum(1 for r in rows if r["o"]),
            "negative": sum(1 for r in rows if not r["o"]),
            "multi_object": sum(1 for r in rows if len(r["o"]) >= 2),
            "instances": dict(sorted(Counter(names[o["c"]] for r in rows for o in r["o"]).items(),
                                     key=lambda kv: names.index(kv[0]))),
        }
    manifest = {
        "names": names,
        "splits": SPLIT_ORDER,
        "summary": per_split,
        "missing_images": len(missing_img),
        "missing_labels": len(missing_lbl),
        "images": images,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    (OUT_DIR / "index.html").write_text(HTML, encoding="utf-8")

    print(f"{'split':<14}{'images':>8}{'pos':>7}{'neg':>6}{'multi':>7}  instances")
    for s, d in per_split.items():
        print(f"{s:<14}{d['images']:>8}{d['positive']:>7}{d['negative']:>6}{d['multi_object']:>7}  "
              + ", ".join(f"{k} {v}" for k, v in d["instances"].items()))
    print(f"train list entries: {sum(oversample.values())} ({len(oversample)} unique)")
    print(f"missing images: {len(missing_img)} | missing labels: {len(missing_lbl)} | bad class ids: {len(bad_cls)}")
    for p in (missing_img + missing_lbl + bad_cls)[:10]:
        print("   ", p)

    # ---- cross-check with the build ----
    ok = not (missing_img or missing_lbl or bad_cls)
    stats_file = DATASET_DIR / "build_9class_stats.json"
    if stats_file.exists():
        st = json.loads(stats_file.read_text())["splits"]
        for ours, theirs in [("train", "train"), ("val", "valid"), ("test", "test")]:
            exp = (st[theirs]["positive"], st[theirs]["negative"])
            got = (per_split[ours]["positive"], per_split[ours]["negative"])
            match = exp == got
            ok &= match
            print(f"build stats {theirs:<6} positive/negative expected {exp} got {got}  {'OK' if match else 'MISMATCH'}")
    print(f"\nWrote {OUT_DIR / 'dataset_manifest.json'} ({(OUT_DIR / 'dataset_manifest.json').stat().st_size / 1e6:.1f} MB)"
          f" and {OUT_DIR / 'index.html'}")
    print("Serve:  python -m http.server 8000 --directory .   ->  http://localhost:8000/docs/viewer/index.html")
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    return 0 if ok else 1


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>9-Class Dataset Viewer</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--fg:#1d2330;--muted:#6b7280;--line:#e3e6eb;--accent:#2563eb;--chip:#eef1f5}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--panel:#1b1f27;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313c;--accent:#60a5fa;--chip:#252b35}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.app{display:grid;grid-template-columns:280px 1fr;min-height:100vh}
aside{background:var(--panel);border-right:1px solid var(--line);padding:16px;position:sticky;top:0;height:100vh;overflow:auto}
h1{font-size:17px;margin:0 0 4px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line);background:var(--chip);color:var(--fg);border-radius:999px;padding:4px 10px;cursor:pointer;font-size:12px;display:flex;align-items:center;gap:6px}
.chip .dot{width:10px;height:10px;border-radius:50%}
.chip.on{background:var(--chip);color:var(--fg);border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.chip:not(.on){opacity:.45}
.chip .cnt{color:var(--muted)}
label.row{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer}
label.row .cnt{margin-left:auto;color:var(--muted);font-size:12px}
.small{font-size:12px;color:var(--muted)}
.btns{display:flex;gap:6px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--chip);color:var(--fg);border-radius:6px;padding:5px 10px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.4;cursor:default}
input[type=range]{width:100%}
input[type=search],select{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
main{padding:16px;min-width:0}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.top .grow{flex:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.card:hover{border-color:var(--accent)}
.thumb{position:relative;aspect-ratio:4/3;background:#000}
.thumb img,.thumb canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.meta{padding:6px 8px;font-size:12px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.tag{padding:1px 6px;border-radius:4px;background:var(--chip);color:var(--muted);font-size:11px}
.fname{width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}
.pager{display:flex;align-items:center;gap:8px;justify-content:center;margin:16px 0}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;z-index:10}
.modal.open{display:grid;grid-template-columns:1fr 320px}
.stage{position:relative;overflow:hidden;cursor:grab}
.stage.drag{cursor:grabbing}
.stage canvas{position:absolute;left:0;top:0}
.side{background:var(--panel);padding:16px;overflow:auto}
.side table{width:100%;border-collapse:collapse;font-size:12px}
.side td,.side th{padding:4px;border-bottom:1px solid var(--line);text-align:left}
.side tr.hl{background:var(--chip)}
.close{position:absolute;top:10px;right:10px;z-index:2}
.navbtn{position:absolute;top:50%;z-index:2;transform:translateY(-50%)}
.empty{padding:40px;text-align:center;color:var(--muted)}
@media (max-width:800px){.app{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
 .modal.open{grid-template-columns:1fr;grid-template-rows:1fr 40%}}
</style>
</head>
<body>
<div class="app">
<aside>
  <h1>9-Class Dataset Viewer</h1>
  <div class="small" id="sumline">Loading manifest...</div>

  <h2>Classes <span class="small">(any of)</span></h2>
  <div class="chips" id="classChips"></div>
  <div class="btns" style="margin-top:8px"><button id="clsAll">All</button><button id="clsNone">None</button></div>

  <h2>Splits</h2>
  <div id="splitBoxes"></div>

  <h2>Filters</h2>
  <label class="row"><input type="checkbox" id="multi"> 2+ target objects</label>
  <label class="row">Largest box area %
    <select id="areaSel" style="width:auto;margin-left:auto">
      <option value="all">any</option><option value="small">&lt; 5% (small)</option>
      <option value="mid">5–50%</option><option value="large">50–80%</option><option value="close">&gt; 80% (close-up)</option>
    </select></label>
  <input type="search" id="q" placeholder="Filename contains..." style="margin-top:6px">
  <select id="sortSel" style="margin-top:6px">
    <option value="name">Sort: split, filename</option><option value="objs">Sort: most objects</option>
    <option value="areaDesc">Sort: largest box first</option><option value="areaAsc">Sort: smallest box first</option>
  </select>

  <h2>Overlay</h2>
  <div class="btns"><button id="tMask" class="on">Show Masks</button><button id="tBox" class="on">Show Boxes</button><button id="tLbl" class="on">Labels</button></div>
  <div style="margin-top:10px" class="small">Mask opacity <span id="opv">35%</span></div>
  <input type="range" id="op" min="0" max="100" value="35">
</aside>

<main>
  <div class="top"><div class="grow small" id="countLine"></div>
    <select id="perPage" style="width:auto"><option>20</option><option selected>40</option><option>80</option><option>160</option></select></div>
  <div class="grid" id="grid"></div>
  <div class="pager"><button id="prev">&larr; Prev</button><span id="pageLbl" class="small"></span><button id="next">Next &rarr;</button></div>
</main>
</div>

<div class="modal" id="modal">
  <div class="stage" id="stage">
    <button class="close" id="mClose">Close (Esc)</button>
    <button class="navbtn" style="left:10px" id="mPrev">&larr;</button>
    <button class="navbtn" style="right:10px" id="mNext">&rarr;</button>
    <canvas id="mCanvas"></canvas>
  </div>
  <div class="side" id="mSide"></div>
</div>

<script>
const PALETTE=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#ff2d2d"];
const SPLIT_LABEL={train:"Train",highangle:"High-Angle",hard_negative:"Hard negatives",val:"Val",test:"Test",cctv:"CCTV Benchmark"};
const S={data:null,cls:new Set(),neg:true,splits:new Set(),multi:false,area:"all",q:"",sort:"name",
  mask:true,box:true,lbl:true,op:.35,page:0,per:40,list:[],modalIdx:-1};
const $=id=>document.getElementById(id);
const imgCache=new Map();

function loadImg(src){
  if(imgCache.has(src))return imgCache.get(src);
  const p=new Promise((res,rej)=>{const im=new Image();im.decoding="async";im.onload=()=>res(im);im.onerror=rej;im.src=src;});
  imgCache.set(src,p);if(imgCache.size>400)imgCache.delete(imgCache.keys().next().value);return p;
}
function hexA(h,a){const n=parseInt(h.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`;}
const maxArea=r=>r.o.length?Math.max(...r.o.map(o=>o.a)):0;

// draw objects of record r onto ctx, image occupies (ox,oy,dw,dh); hl = highlighted object index
function drawObjs(ctx,objs,ox,oy,dw,dh,lw,hl){
  const names=S.data.names;
  objs.forEach((o,i)=>{
    const col=PALETTE[o.c%PALETTE.length],p=o.p,dim=hl>=0&&hl!==i;
    if(S.mask){
      ctx.beginPath();
      for(let k=0;k<p.length;k+=2){const x=ox+p[k]*dw,y=oy+p[k+1]*dh;k?ctx.lineTo(x,y):ctx.moveTo(x,y);}
      ctx.closePath();ctx.fillStyle=hexA(col,dim?S.op*.3:S.op);ctx.fill();
      ctx.lineWidth=lw;ctx.strokeStyle=hexA(col,dim?.3:.95);ctx.stroke();
    }
    if(S.box){
      const [x1,y1,x2,y2]=o.b;ctx.lineWidth=hl===i?lw*2:lw;ctx.strokeStyle=dim?hexA(col,.3):col;
      ctx.strokeRect(ox+x1*dw,oy+y1*dh,(x2-x1)*dw,(y2-y1)*dh);
    }
    if(S.lbl&&(S.box||S.mask)&&!dim){
      const fs=Math.max(10,lw*6),t=names[o.c].replace(/_/g," ");
      ctx.font=`600 ${fs}px system-ui,sans-serif`;
      const tw=ctx.measureText(t).width+6,x=ox+o.b[0]*dw,y=Math.max(oy+o.b[1]*dh,oy+fs+4);
      ctx.fillStyle=col;ctx.fillRect(x,y-fs-4,tw,fs+4);ctx.fillStyle="#fff";ctx.fillText(t,x+3,y-4);
    }
  });
}

function renderCard(card){
  const r=card._r,cv=card.querySelector("canvas"),im=card.querySelector("img");
  if(!im.complete||!im.naturalWidth)return;
  const W=cv.clientWidth,H=cv.clientHeight,dpr=window.devicePixelRatio||1;
  cv.width=W*dpr;cv.height=H*dpr;const ctx=cv.getContext("2d");ctx.scale(dpr,dpr);
  const s=Math.min(W/r.w,H/r.h),dw=r.w*s,dh=r.h*s;
  drawObjs(ctx,r.o,(W-dw)/2,(H-dh)/2,dw,dh,1.5,-1);
}

const io=new IntersectionObserver(es=>es.forEach(e=>{
  if(!e.isIntersecting)return;const card=e.target;io.unobserve(card);
  const im=card.querySelector("img");im.onload=()=>renderCard(card);im.src=card._r.f;
}),{rootMargin:"300px"});

function applyFilters(){
  const d=S.data;
  let L=d.images.filter(r=>{
    if(!S.splits.has(r.s))return false;
    if(r.o.length){if(!r.o.some(o=>S.cls.has(o.c)))return false;}else if(!S.neg)return false;
    if(S.multi&&r.o.length<2)return false;
    if(S.area!=="all"){const a=maxArea(r);
      if(!r.o.length)return false;
      if(S.area==="small"&&a>=5)return false;if(S.area==="mid"&&(a<5||a>=50))return false;
      if(S.area==="large"&&(a<50||a>=80))return false;if(S.area==="close"&&a<80)return false;}
    if(S.q&&!r.f.toLowerCase().includes(S.q))return false;
    return true;
  });
  if(S.sort==="objs")L=[...L].sort((a,b)=>b.o.length-a.o.length);
  if(S.sort==="areaDesc")L=[...L].sort((a,b)=>maxArea(b)-maxArea(a));
  if(S.sort==="areaAsc")L=[...L].sort((a,b)=>(a.o.length?maxArea(a):1e9)-(b.o.length?maxArea(b):1e9));
  S.list=L;S.page=Math.min(S.page,Math.max(0,Math.ceil(L.length/S.per)-1));
  const inst=L.reduce((n,r)=>n+r.o.length,0);
  $("countLine").textContent=`${L.length.toLocaleString()} images · ${inst.toLocaleString()} instances match`;
  renderGrid();
}

function renderGrid(){
  const g=$("grid");g.innerHTML="";io.disconnect();
  const pages=Math.max(1,Math.ceil(S.list.length/S.per)),start=S.page*S.per;
  const slice=S.list.slice(start,start+S.per);
  if(!slice.length)g.innerHTML='<div class="empty">No images match these filters.</div>';
  slice.forEach((r,i)=>{
    const c=document.createElement("div");c.className="card";c._r=r;
    const name=r.f.split("/").pop();
    const cls=[...new Set(r.o.map(o=>o.c))];
    c.innerHTML=`<div class="thumb"><img alt=""><canvas></canvas></div><div class="meta">
      <span class="tag">${SPLIT_LABEL[r.s]}</span>
      ${r.o.length?cls.map(k=>`<span class="tag" style="color:${PALETTE[k]}">${S.data.names[k]}</span>`).join(""):'<span class="tag">negative</span>'}
      ${r.o.length>1?`<span class="tag">${r.o.length} obj</span>`:""}${r.n>1?`<span class="tag" title="entries in the oversampled train list">×${r.n}</span>`:""}
      <div class="fname" title="${name}">${name}</div></div>`;
    c.onclick=()=>openModal(start+i);g.appendChild(c);io.observe(c);
  });
  $("pageLbl").textContent=`Page ${S.page+1} / ${pages}`;
  $("prev").disabled=S.page===0;$("next").disabled=S.page>=pages-1;
}
const redrawCards=()=>document.querySelectorAll(".card").forEach(renderCard);

// ---------- modal with zoom / pan ----------
const M={im:null,objs:null,exact:false,z:1,x:0,y:0,hl:-1};
// exact polygons straight from the label file (the manifest holds simplified ones for thumbnails)
async function loadExact(r){
  const txt=await (await fetch(r.l)).text();
  return txt.split(/\r?\n/).filter(s=>s.trim()).map((line,i)=>{
    const v=line.trim().split(/\s+/).map(Number);let p=v.slice(1);
    if(p.length===4){const[cx,cy,w,h]=p;p=[cx-w/2,cy-h/2,cx+w/2,cy-h/2,cx+w/2,cy+h/2,cx-w/2,cy+h/2];}
    return {...r.o[i],c:v[0],p};
  });
}
function fitModal(){const st=$("stage"),r=S.list[S.modalIdx];M.z=Math.min(st.clientWidth/r.w,st.clientHeight/r.h);
  M.x=(st.clientWidth-r.w*M.z)/2;M.y=(st.clientHeight-r.h*M.z)/2;}
function drawModal(){
  const st=$("stage"),cv=$("mCanvas"),r=S.list[S.modalIdx],dpr=window.devicePixelRatio||1;
  cv.width=st.clientWidth*dpr;cv.height=st.clientHeight*dpr;cv.style.width=st.clientWidth+"px";cv.style.height=st.clientHeight+"px";
  const ctx=cv.getContext("2d");ctx.scale(dpr,dpr);ctx.imageSmoothingEnabled=M.z<2;
  if(!M.im)return;const dw=r.w*M.z,dh=r.h*M.z;
  ctx.drawImage(M.im,M.x,M.y,dw,dh);drawObjs(ctx,M.objs||r.o,M.x,M.y,dw,dh,Math.max(1.5,Math.min(3,M.z*2)),M.hl);
}
async function openModal(idx){
  S.modalIdx=idx;const r=S.list[idx];$("modal").classList.add("open");M.im=null;M.objs=null;M.hl=-1;
  const names=S.data.names,name=r.f.split("/").pop();
  history.replaceState(null,"","#img="+encodeURIComponent(name));
  $("mSide").innerHTML=`<h1 style="word-break:break-all;font-size:14px">${name}</h1>
    <div class="small">${SPLIT_LABEL[r.s]} · ${r.w}×${r.h}px${r.n>1?` · ×${r.n} in train list`:""} · ${idx+1}/${S.list.length}</div>
    <div class="small" style="margin:6px 0;word-break:break-all"><a href="${r.f}" target="_blank" style="color:var(--accent)">open original</a> · ${r.f.replace(/^(\.\.\/)+/,"")}</div>
    ${r.o.length?`<table><tr><th>#</th><th>Class</th><th>Vertices</th><th>Box %</th></tr>${r.o.map((o,i)=>
      `<tr data-i="${i}"><td>${i+1}</td><td><span style="color:${PALETTE[o.c]}">■</span> ${names[o.c]}</td><td>${o.v}</td><td>${o.a.toFixed(1)}</td></tr>`).join("")}</table>
      <div class="small" id="exactNote" style="margin-top:8px">Loading exact polygons from the label file...</div>
      <div class="small" style="margin-top:8px">Hover a row to highlight it. Scroll to zoom, drag to pan, double-click to fit. M / B toggle masks / boxes, ← → navigate.</div>`
      :'<p class="small">Negative / background: empty label file.</p>'}`;
  $("mSide").querySelectorAll("tr[data-i]").forEach(tr=>{tr.onmouseenter=()=>{M.hl=+tr.dataset.i;drawModal();};tr.onmouseleave=()=>{M.hl=-1;drawModal();};});
  fitModal();drawModal();
  if(r.o.length)loadExact(r).then(objs=>{if(S.modalIdx!==idx)return;M.objs=objs;drawModal();
    const n=$("exactNote");if(n)n.textContent="Showing exact label polygons (thumbnails use a 1 px simplification).";})
    .catch(()=>{const n=$("exactNote");if(n)n.textContent="Label file not reachable; showing the simplified thumbnail polygons.";});
  try{const im=await loadImg(r.f);if(S.modalIdx===idx){M.im=im;drawModal();}}catch(e){}
}
function closeModal(){$("modal").classList.remove("open");S.modalIdx=-1;history.replaceState(null,"",location.pathname);}
// deep link: index.html#img=<filename> opens that image (searches all splits, ignoring filters)
function openFromHash(){
  const m=location.hash.match(/^#img=(.+)$/);if(!m)return;const name=decodeURIComponent(m[1]);
  let i=S.list.findIndex(r=>r.f.endsWith("/"+name));
  if(i<0){const r=S.data.images.find(r=>r.f.endsWith("/"+name));if(!r)return;S.list=[r];renderGrid();i=0;}
  S.page=Math.floor(i/S.per);renderGrid();openModal(i);
}
function stepModal(d){if(S.modalIdx<0)return;const n=S.modalIdx+d;if(n<0||n>=S.list.length)return;
  const p=Math.floor(n/S.per);if(p!==S.page){S.page=p;renderGrid();}openModal(n);}
(function(){
  const st=$("stage");let drag=null;
  st.addEventListener("wheel",e=>{if(S.modalIdx<0)return;e.preventDefault();const rect=st.getBoundingClientRect();
    const mx=e.clientX-rect.left,my=e.clientY-rect.top,f=Math.exp(-e.deltaY*0.0015),nz=Math.min(40,Math.max(0.05,M.z*f));
    M.x=mx-(mx-M.x)*nz/M.z;M.y=my-(my-M.y)*nz/M.z;M.z=nz;drawModal();},{passive:false});
  st.addEventListener("mousedown",e=>{if(e.target.tagName==="BUTTON")return;drag={x:e.clientX,y:e.clientY,ox:M.x,oy:M.y};st.classList.add("drag");});
  window.addEventListener("mousemove",e=>{if(!drag)return;M.x=drag.ox+e.clientX-drag.x;M.y=drag.oy+e.clientY-drag.y;drawModal();});
  window.addEventListener("mouseup",()=>{drag=null;st.classList.remove("drag");});
  st.addEventListener("dblclick",()=>{fitModal();drawModal();});
  window.addEventListener("resize",()=>{if(S.modalIdx>=0){fitModal();drawModal();}else redrawCards();});
})();
$("mClose").onclick=closeModal;$("mPrev").onclick=()=>stepModal(-1);$("mNext").onclick=()=>stepModal(1);
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"&&e.target.type==="search")return;
  if(e.key==="Escape")closeModal();
  if(S.modalIdx>=0&&e.key==="ArrowRight")stepModal(1);
  if(S.modalIdx>=0&&e.key==="ArrowLeft")stepModal(-1);
  if(e.key==="m"||e.key==="M")$("tMask").click();
  if(e.key==="b"||e.key==="B")$("tBox").click();
});

// ---------- controls ----------
function buildControls(){
  const d=S.data,names=d.names,imgs=d.images;
  const cc=new Array(names.length).fill(0);let negN=0;
  imgs.forEach(r=>{if(!r.o.length)negN++;new Set(r.o.map(o=>o.c)).forEach(c=>cc[c]++);});
  const chips=$("classChips");
  const mk=(label,color,cnt,on,toggle)=>{const b=document.createElement("button");b.className="chip"+(on?" on":"");
    b.innerHTML=`<span class="dot" style="background:${color}"></span>${label}<span class="cnt">${cnt}</span>`;
    b.onclick=()=>{b.classList.toggle("on",toggle());S.page=0;applyFilters();};chips.appendChild(b);return b;};
  const clsBtns=names.map((n,i)=>{S.cls.add(i);return mk(n.replace(/_/g," "),PALETTE[i],cc[i],true,()=>{S.cls.has(i)?S.cls.delete(i):S.cls.add(i);return S.cls.has(i);});});
  const negBtn=mk("negatives / background","#888",negN,true,()=>(S.neg=!S.neg));
  const setAll=on=>{names.forEach((_,i)=>on?S.cls.add(i):S.cls.delete(i));S.neg=on;
    clsBtns.forEach(b=>b.classList.toggle("on",on));negBtn.classList.toggle("on",on);S.page=0;applyFilters();};
  $("clsAll").onclick=()=>setAll(true);$("clsNone").onclick=()=>setAll(false);

  const sb=$("splitBoxes");
  d.splits.forEach(s=>{const n=imgs.filter(r=>r.s===s).length;if(!n)return;
    const on=s!=="cctv";if(on)S.splits.add(s);
    const l=document.createElement("label");l.className="row";
    l.innerHTML=`<input type="checkbox" ${on?"checked":""}> ${SPLIT_LABEL[s]}<span class="cnt">${n}</span>`;
    l.querySelector("input").onchange=e=>{e.target.checked?S.splits.add(s):S.splits.delete(s);S.page=0;applyFilters();};sb.appendChild(l);});

  $("multi").onchange=e=>{S.multi=e.target.checked;S.page=0;applyFilters();};
  $("areaSel").onchange=e=>{S.area=e.target.value;S.page=0;applyFilters();};
  $("sortSel").onchange=e=>{S.sort=e.target.value;S.page=0;applyFilters();};
  let qt;$("q").oninput=e=>{clearTimeout(qt);qt=setTimeout(()=>{S.q=e.target.value.trim().toLowerCase();S.page=0;applyFilters();},200);};
  $("perPage").onchange=e=>{S.per=+e.target.value;S.page=0;applyFilters();};
  $("prev").onclick=()=>{S.page--;renderGrid();scrollTo(0,0);};$("next").onclick=()=>{S.page++;renderGrid();scrollTo(0,0);};
  const tog=(id,key)=>{$(id).onclick=()=>{S[key]=!S[key];$(id).classList.toggle("on",S[key]);redrawCards();if(S.modalIdx>=0)drawModal();};};
  tog("tMask","mask");tog("tBox","box");tog("tLbl","lbl");
  $("op").oninput=e=>{S.op=e.target.value/100;$("opv").textContent=e.target.value+"%";redrawCards();if(S.modalIdx>=0)drawModal();};

  const inst=imgs.reduce((n,r)=>n+r.o.length,0);
  $("sumline").textContent=`${imgs.length.toLocaleString()} images · ${inst.toLocaleString()} instances · missing labels: ${d.missing_labels}`;
}

fetch("dataset_manifest.json").then(r=>r.json()).then(d=>{S.data=d;buildControls();applyFilters();openFromHash();})
  .catch(e=>{$("sumline").textContent="Could not load dataset_manifest.json. Serve the project root over HTTP (python -m http.server 8000 --directory .), do not open the file directly.";});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
