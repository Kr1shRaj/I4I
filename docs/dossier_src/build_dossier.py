"""Build the Krish technical dossier (v2.6: no table of contents).

    env\\yolo-env\\Scripts\\python.exe docs\\dossier_src\\build_dossier.py . docs\\Krish_Technical_Dossier_v2.6.docx

Needs python-docx (pure Python; lxml is already in the venv). Older versions' sources are in v2.5\\.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dossier_lib import Dossier
import dossier_part1 as p1
import dossier_part2 as p2

CHAPTERS = {"1": p1.ch1, "2": p1.ch2, "3": p1.ch3, "4": p2.ch4, "5": None, "6": p2.ch6, "7": p2.ch7, "8": p2.ch8}


def build(root, out, chapters="12345678", title=True):
    d = Dossier()
    d._root = root
    if title:
        p1.front(d)
    for c in chapters:
        p2.ch5(d, root) if c == "5" else CHAPTERS[c](d)
    d.save(out)
    return d


if __name__ == "__main__":
    ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "Krish_Technical_Dossier_v2.6.docx"
    d = build(ROOT, OUT)
    print(f"saved {OUT} | tables: {d.tab} | figures: {d.fig}")
