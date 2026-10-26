"""
Parse the user's curation file (`datasetchange` at the project root) into clean, located stems.

Sections: "TO FIX ANNOTATIONS" and "TO REMOVE FROM DATASET...". Each line is an image filename; whitespace,
extensions and "Img_" casing are normalised and duplicates dropped. Each stem is located in
datasets/source_38class/<split>/images and its current 9-class build status is reported.

Output: datasets/9class/label_fixes/user_curation.json
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "datasets" / "source_38class"
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
FIX_DIR = DATASET_DIR / "label_fixes"
TARGET_MAP = {14: 0, 5: 1, 11: 2, 15: 3, 16: 4, 12: 5, 13: 6, 26: 7, 21: 8}


def clean(line):
    s = line.strip()
    if not s:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".txt"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    if s.lower().startswith("img_"):
        s = "img_" + s[4:]
    return s


def parse(path):
    sections, cur = {"fix": [], "remove": []}, None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        u = line.strip().upper()
        if u.startswith("TO FIX"):
            cur = "fix"
            continue
        if u.startswith("TO REMOVE"):
            cur = "remove"
            continue
        s = clean(line)
        if s and cur:
            sections[cur].append(s)
    return sections


def build_status():
    """Which stems the current build lists, per split (from the generated lists)."""
    listed = {}
    for split, name in (("train", "train_oversampled_9class.txt"), ("valid", "valid_9class.txt"),
                        ("test", "test_9class.txt")):
        for line in (DATASET_DIR / name).read_text().split():
            listed.setdefault(Path(line).stem, split)
    return listed


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "datasetchange"
    raw = parse(src)
    listed = build_status()
    dec = yaml.safe_load((FIX_DIR / "label_decisions.yaml").read_text(encoding="utf-8"))
    eb_props = json.loads((FIX_DIR / "edge_bander_proposals.json").read_text(encoding="utf-8"))
    eb_q = {eb_props[i]["stem"] for i in dec["edge_bander"]["quarantine"]}
    fe_q = set(dec["fire_extinguisher"]["quarantine"] or [])
    cad = set(json.loads((DATASET_DIR / "quarantined_diagrams_54.json").read_text())["stems"])

    out = {"source": str(src.relative_to(PROJECT_ROOT)), "raw_lines": {k: len(v) for k, v in raw.items()}}
    for key, stems in raw.items():
        uniq = list(dict.fromkeys(stems))
        recs, missing = [], []
        for s in uniq:
            split = next((sp for sp in ("train", "valid", "test")
                          if (SRC_DIR / sp / "images" / (s + ".jpg")).exists()), None)
            if split is None:
                missing.append(s)
                continue
            lp = SRC_DIR / split / "labels" / (s + ".txt")
            classes = sorted({TARGET_MAP[int(l.split()[0])] for l in lp.read_text().splitlines()
                              if l.strip() and int(l.split()[0]) in TARGET_MAP}) if lp.exists() else []
            status = ("listed" if s in listed else "cad_quarantine" if s in cad
                      else "label_fix_quarantine" if s in eb_q or s in fe_q else "not_in_build")
            recs.append({"stem": s, "split": split, "classes": classes, "status": status})
        out[key] = {"lines": len(stems), "unique": len(uniq), "duplicates": len(stems) - len(uniq),
                    "missing": missing, "images": recs}
    both = {r["stem"] for r in out["fix"]["images"]} & {r["stem"] for r in out["remove"]["images"]}
    out["in_both_lists"] = sorted(both)

    (FIX_DIR / "user_curation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ("fix", "remove"):
        o = out[key]
        print(f"[{key}] lines {o['lines']} | unique {o['unique']} | duplicates {o['duplicates']} | "
              f"missing {len(o['missing'])}")
        for k in ("train", "valid", "test"):
            n = sum(r["split"] == k for r in o["images"])
            print(f"   {k}: {n}")
        st = {}
        for r in o["images"]:
            st[r["status"]] = st.get(r["status"], 0) + 1
        print(f"   status: {st}")
        for m in o["missing"]:
            print(f"   MISSING: {m}")
    print(f"in both lists: {len(both)} {sorted(both)}")


if __name__ == "__main__":
    main()
