"""Build the Krish technical dossier.

    env\\yolo-env\\Scripts\\python.exe docs\\dossier_src\\build_dossier.py . docs\\Krish_Technical_Dossier_v2.5.docx

Needs python-docx (pure Python; lxml is already in the venv).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dossier_lib import Dossier
import dossier_part1 as p1
import dossier_part2 as p2

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "Krish_Technical_Dossier_v2.5.docx"

d = Dossier()
d._root = ROOT
p1.front(d)
p1.ch1(d)
p1.ch2(d)
p1.ch3(d)
p2.ch4(d)
p2.ch5(d, ROOT)
p2.ch6(d)
p2.ch7(d)
p2.ch8(d)
p2.ch9(d)
p2.ch10(d)
p2.appendix(d)
d.save(OUT)
print(f"saved {OUT} | headings in TOC: {len(d.toc)} | tables: {d.tab} | figures: {d.fig}")
