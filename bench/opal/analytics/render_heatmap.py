#!/usr/bin/env python3
"""
Render an attn_analytics.py CSV as a text heat map.

One column per reply segment. For each line of the prompt the cell is the
maximum score over that line's tokens, scaled to its own column's maximum and
drawn on the ramp ' .:-=+*#%@' (level 0..9). A blank means level 0, which is
either no attention or no token on that line surviving the connector's
top-`top_pct` selection.

Scaling is per column, so columns show where an item looked, not how much
attention it had relative to another item.

The CSV is self-contained -- it carries `model` and `prompt_token_ids` -- so
this needs no chain file, no GPU and no vLLM, only the tokenizer.

Usage
-----
    render_heatmap.py attn_qwen.csv --round 5
    render_heatmap.py attn_qwen.csv --round 5 --width 110 > round5.txt
    render_heatmap.py attn_qwen.csv --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAMP = " .:-=+*#%@"


def line_index(ids: list[int], tok) -> tuple[list[str], np.ndarray]:
    """Prompt lines, and the line each token belongs to."""
    lines = tok.decode(ids).split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    line_of, ln = np.zeros(len(ids), dtype=int), 0
    for j, t in enumerate(ids):
        line_of[j] = min(ln, len(lines) - 1)
        ln += tok.decode([int(t)]).count("\n")
    return lines, line_of


def render(rows: list[dict], tok, width: int) -> str:
    ids = [int(x) for x in rows[0]["prompt_token_ids"].split()]
    lines, line_of = line_index(ids, tok)

    cols = []
    for r in rows:
        sc = np.fromstring(r["scores"], sep=" ")
        if len(sc) != len(ids):
            sys.exit(f"error: round {r['round_no']} segment {r['segment_index']}: "
                     f"{len(sc)} scores for {len(ids)} tokens")
        per = np.zeros(len(lines))
        np.maximum.at(per, line_of, sc)
        m = per.max() or 1.0
        cols.append([min(9, int(round(9 * x / m))) for x in per])

    head = [f"{rows[0]['chain_id']} round {rows[0]['round_no']} | "
            f"{rows[0]['task_type']} | {rows[0]['model']}"]
    for i, r in enumerate(rows):
        head.append(f"  col{i + 1} = segment {r['segment_index']} "
                    f"{r['segment_text']!r}  (decode steps {r['decode_steps']})")
    head += [f"  ramp {RAMP!r}, level 0..9, each column scaled to its own max",
             f"  {len(ids)} tokens, {len(lines)} lines", "=" * (width + len(cols) + 1)]

    body = ["".join(RAMP[c[i]] for c in cols) + " " + line[:width]
            for i, line in enumerate(lines)]
    return "\n".join(head + body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--round", type=int, help="round_no to render")
    ap.add_argument("--all", action="store_true", help="render every round")
    ap.add_argument("--width", type=int, default=88, help="prompt columns to show")
    args = ap.parse_args(argv)
    if not args.round and not args.all:
        ap.error("give --round N or --all")

    csv.field_size_limit(1 << 30)
    by_round: dict[int, list[dict]] = defaultdict(list)
    for r in csv.DictReader(args.csv.open()):
        by_round[int(r["round_no"])].append(r)
    if not by_round:
        sys.exit(f"error: no rows in {args.csv}")

    missing = {"model", "prompt_token_ids"} - set(next(iter(by_round.values()))[0])
    if missing:
        sys.exit(f"error: {args.csv} predates the self-contained format, missing "
                 f"{sorted(missing)}; regenerate it with attn_analytics.py")

    from transformers import AutoTokenizer

    wanted = sorted(by_round) if args.all else [args.round]
    toks: dict[str, object] = {}
    for n in wanted:
        if n not in by_round:
            sys.exit(f"error: round {n} not in {args.csv}; have {sorted(by_round)}")
        rows = sorted(by_round[n], key=lambda r: int(r["segment_index"]))
        name = rows[0]["model"]
        if name not in toks:
            toks[name] = AutoTokenizer.from_pretrained(name)
        print(render(rows, toks[name], args.width))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
