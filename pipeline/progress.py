"""
pipeline/progress.py
====================
Append-only progress events for the dataset UI's pipeline monitor: logs/progress.jsonl, one JSON object per line.

Wrap a stage's entry point, then report per-class progress from its loop:

    from pipeline import progress

    if __name__ == "__main__":
        with progress.run("stage3_clean"):
            main()

    progress.class_start(cls_name, i, len(target_names))
    progress.tick(done, total)                    # optional, inside long per-class work (rate-limited)
    progress.class_end(cls_name, final_count=..)  # counts shown by the monitor

Events: start, class_start, tick, class_end, end (status ok / error / cancelled).
Every line carries t (UTC ISO time), run (id), stage and pid. Writing never raises: a full disk or a locked file
must not break a scrape. Outside `run()` the functions do nothing, so importing stages from run.py is safe.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOGS_DIR

PROGRESS_FILE = Path(os.environ.get("PIPELINE_PROGRESS_FILE") or LOGS_DIR / "progress.jsonl")   # env: tests
TICK_INTERVAL_S = 1.0
_stack: list[dict] = []   # nested runs (run.py driving several stages); events go to the innermost


def _emit(event: str, run: dict | None = None, **fields) -> None:
    run = run or (_stack[-1] if _stack else None)
    if run is None:
        return
    rec = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event,
           "run": run["id"], "stage": run["stage"], "pid": os.getpid(), **fields}
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


@contextmanager
def run(stage: str, **params):
    """Bracket one stage run with start/end events; the end event records ok, error or cancelled."""
    rec = {"id": f"{stage}-{os.getpid()}-{int(time.time() * 1000)}", "stage": stage, "t0": time.monotonic(),
           "last_tick": 0.0}
    _emit("start", rec, argv=sys.argv[1:], params=params, parent=_stack[-1]["id"] if _stack else None)
    _stack.append(rec)
    status, extra = "ok", {}
    try:
        yield rec
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    except SystemExit as e:
        if e.code not in (None, 0):
            status, extra = "error", {"error": f"exit code {e.code}"}
        raise
    except BaseException as e:
        status = "error"
        extra = {"error": f"{type(e).__name__}: {e}",
                 "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]}
        raise
    finally:
        _stack.pop()
        _emit("end", rec, status=status, seconds=round(time.monotonic() - rec["t0"], 1), **extra)


def class_start(name: str, index: int | None = None, total: int | None = None) -> None:
    _emit("class_start", cls=name, index=index, total=total)


def tick(done: int, total: int | None = None, **info) -> None:
    """Progress inside one class; written at most once per TICK_INTERVAL_S (always when done == total)."""
    if not _stack:
        return
    now = time.monotonic()
    if total is not None and done >= total or now - _stack[-1]["last_tick"] >= TICK_INTERVAL_S:
        _stack[-1]["last_tick"] = now
        _emit("tick", done=done, total=total, **info)


def class_end(name: str, **counts) -> None:
    _emit("class_end", cls=name, **counts)
