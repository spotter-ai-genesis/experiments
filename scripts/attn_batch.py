#!/usr/bin/env python
"""Reduce every run under runs/ to analytics CSVs and text heat maps.

    python scripts/attn_batch.py                    # every run
    python scripts/attn_batch.py --runs runs/clean  # just these
    python scripts/attn_batch.py --no-render        # CSVs only

For each run this writes, under `analytics-out/<run name>/`:

    <run>.csv              every chain, one row per (round, reply item)
    chains/<chain_id>.csv  the same rows split by chain
    renders/<chain_id>.txt every round of that chain as a heat map

The split is a filter on the combined CSV, not a second reduction, so a
per-chain file is byte-identical to the rows it came from. Rendering needs one
chain per file because render_heatmap.py keys the prompt off the first row of
a round, which only holds for rows that share a chain.

A run is eligible when its turns carry `gen_char_ends` and its capture is
non-zero. Anything else is listed as skipped rather than silently dropped --
an all-zero capture is not self-announcing, and five runs were once reduced,
plotted and believed before anyone noticed there was no attention in them.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent          # scripts/
ROOT = HERE.parent                              # repo root
ANALYTICS = ROOT / "bench" / "opal" / "analytics"
RUNS = ROOT / "runs"
OUT = ROOT / "analytics-out"


def eligible(run: Path) -> str | None:
    """Reason this run cannot be reduced, or None if it can."""
    tj = run / "turns.jsonl"
    if not tj.is_file():
        return "no turns.jsonl"
    turns = [json.loads(x) for x in tj.read_text().splitlines() if x.strip()]
    if not turns:
        return "no turns"
    if not turns[0].get("gen_char_ends"):
        return "turns carry no gen_char_ends"
    meta_p = run / "run_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    if not meta.get("probe_layers_seen"):
        return "capture is empty"
    return None


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ANALYTICS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_and_render(combined: Path, render, toks) -> tuple[int, int]:
    csv.field_size_limit(1 << 30)
    rows = list(csv.DictReader(combined.open()))
    if not rows:
        return 0, 0
    by_chain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chain[r["chain_id"]].append(r)

    cdir, rdir = combined.parent / "chains", combined.parent / "renders"
    cdir.mkdir(exist_ok=True)
    if render:
        rdir.mkdir(exist_ok=True)

    for chain, crows in sorted(by_chain.items()):
        with (cdir / f"{chain}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(crows)
        if not render:
            continue
        by_round: dict[int, list[dict]] = defaultdict(list)
        for r in crows:
            by_round[int(r["round_no"])].append(r)
        out = []
        for n in sorted(by_round):
            rr = sorted(by_round[n], key=lambda r: int(r["segment_index"]))
            name = rr[0]["model"]
            if name not in toks:
                from transformers import AutoTokenizer
                toks[name] = AutoTokenizer.from_pretrained(name)
            out.append(render.render(rr, toks[name], 88))
        (rdir / f"{chain}.txt").write_text("\n\n".join(out) + "\n")
    return len(rows), len(by_chain)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", type=Path,
                    help=f"default: every directory under {RUNS.name}/")
    ap.add_argument("--no-render", action="store_true",
                    help="CSVs only, skip the heat maps (skips the tokenizer)")
    args = ap.parse_args(argv)

    runs = args.runs or sorted(d for d in RUNS.glob("*") if d.is_dir())
    if not runs:
        sys.exit(f"error: no runs under {RUNS}. Run scripts/run_suite.sh first.")
    render = None if args.no_render else _load("render_heatmap")
    toks: dict[str, object] = {}

    done, skipped = [], []
    for run in runs:
        why = eligible(run)
        if why:
            skipped.append((run.name, why))
            continue
        d = OUT / run.name
        d.mkdir(parents=True, exist_ok=True)
        combined = d / f"{run.name}.csv"
        print(f"== {run.name}", flush=True)
        r = subprocess.run(
            [sys.executable, str(HERE / "attn_from_run.py"),
             str(run), "--out", str(combined)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            sys.exit(f"error: reduction failed for {run.name}")
        print("   " + r.stdout.strip().splitlines()[-1])
        n_rows, n_chains = split_and_render(combined, render, toks)
        print(f"   split {n_rows} rows into {n_chains} chains"
              + ("" if args.no_render else " and rendered each"), flush=True)
        done.append((run.name, n_rows, n_chains))

    print(f"\nreduced {len(done)} runs -> {OUT}")
    for name, n, c in done:
        print(f"  {name:24} {n:>5} rows  {c:>3} chains")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for name, why in skipped:
            print(f"  {name:24} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
