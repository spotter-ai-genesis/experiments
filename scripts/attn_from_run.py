#!/usr/bin/env python
"""Per-reply-segment prefill attention, from a run we already captured.

    python scripts/attn_from_run.py runs/<run> --out out.csv

`bench/opal/analytics/attn_analytics.py` does two things: it runs a
chain on a GPU, and it reduces the capture to one attention vector per reply
item. This does only the second, against run directories this harness has
already produced, and emits the SAME CSV so `render_heatmap.py` reads it
unchanged.

The reduction is not reimplemented. `aggregate()` and `DECAY` are imported from
that script, so the arithmetic cannot drift from it:

    score[j] = sum_i DECAY**i * val_all_max[step_i][j]

with i counted from the item's first step.

ONE ADAPTATION, AND IT IS NOT OPTIONAL. The original anchors the answer at
`^ANSWER:` because its replies are the answer and nothing else. Ours are from a
thinking model: a typical reply is 727 decode steps of which 720 are reasoning,
and `ANSWER:` appears five times -- inside the reasoning and again after
`</think>`. Anchoring at the start would make `body_at` zero and turn the whole
chain of thought into `;`-separated "items". So the body starts at the LAST
`ANSWER:`, matching how the scorer picks an answer, and the reasoning steps are
excluded by default. `--with-reasoning` adds them as one extra segment, which
is where most of the decode actually is.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # repo root
sys.path.insert(0, str(HERE / "src"))

from opal_attn import capture  # noqa: E402

ANSWER_RE = re.compile(r"ANSWER\s*:\s*", re.IGNORECASE)


def _connector_parent() -> Path | None:
    """A directory CONTAINING a vllm-attn-connector checkout, or None.

    attn_analytics.py locates the connector by walking UP from its own file
    looking for a sibling `vllm-attn-connector/`, and exits if it finds none.
    That holds in the genesis workspace, where the connector sits beside the
    experiment, but not when setup_box.sh clones it into <repo>/vendor/ --
    a CHILD of this tree, which an upward walk can never reach.
    """
    for d in (HERE, HERE / "vendor", *HERE.parents):
        if (d / "vllm-attn-connector" / "src" / "vllm_attn_connector").is_dir():
            return d
    return None


def _load_analytics():
    """Import attn_analytics by path so its reduction is reused, not copied."""
    cand = HERE / "bench" / "opal" / "analytics" / "attn_analytics.py"
    if not cand.is_file():
        sys.exit(f"error: {cand} not found")

    spec = importlib.util.spec_from_file_location("_attn_analytics", cand)
    mod = importlib.util.module_from_spec(spec)

    # Make its upward search succeed by telling it where it lives, rather than
    # editing it or symlinking into the repo. The file is kept byte-identical
    # to upstream on purpose, so the reduction cannot drift; __file__ is the
    # one input to that search and it is ours to set.
    parent = _connector_parent()
    if parent is not None:
        mod.__file__ = str(parent / "_attn_analytics_shim" / cand.name)

    sys.modules["_attn_analytics"] = mod
    try:
        spec.loader.exec_module(mod)          # no vllm/flowcept at import time
    except SystemExit as e:
        sys.exit(f"error loading {cand}: {e}\n"
                 f"  It locates vllm-attn-connector by walking up from its own\n"
                 f"  file. Searched for a checkout under: {HERE}, {HERE / 'vendor'},\n"
                 f"  and the parents of {HERE}. Point OPAL_CONNECTOR_DIR at it, or\n"
                 f"  re-run scripts/setup_box.sh, which clones it into vendor/.")
    return mod, cand


def pieces_of(turn: dict) -> list[str]:
    """Per-decode-step text, from the char offsets the runner recorded.

    Equivalent to the original's `[tok.decode([t]) for t in out_ids]` without
    needing the tokenizer or the ids: gen_char_ends[i] is where step i ends in
    gen_text.
    """
    g, ends = turn.get("gen_text") or "", turn.get("gen_char_ends") or []
    return [g[(ends[i - 1] if i else 0):ends[i]] for i in range(len(ends))]


def segment_steps(pieces: list[str]) -> tuple[list[tuple[str, list[int]]], list[int]]:
    """`([(label, steps)], reasoning_steps)`, splitting the answer on ';'.

    Mirrors attn_analytics.segment_steps, with the body anchored at the LAST
    `ANSWER:` rather than the first character. Empty pieces stand in for the
    special tokens the original skips by id.

    The search is confined to the text AFTER `</think>`, for the same reason
    the scorer confines it: ANSWER_RE is case-insensitive, so a reasoning
    block offers up the model's own prose. One outfmt turn reasoned past
    "Round 1 Answer: `100`" and then replied without the prefix; anchoring on
    that match turned 4.7k characters of reasoning into one "item". With no
    match in the visible reply there is no answer to segment, and the caller
    drops the turn.
    """
    text = "".join(pieces)
    close = text.rfind("</think>")
    visible_at = close + len("</think>") if close >= 0 else 0
    hits = [h for h in ANSWER_RE.finditer(text) if h.start() >= visible_at]
    if not hits:
        return [], list(range(len(pieces)))
    body_at = hits[-1].end()

    start, char = [], 0
    for p in pieces:
        start.append(char)
        char += len(p)

    items: list[list[int]] = []
    reasoning: list[int] = []
    cur: list[int] = []
    for i, piece in enumerate(pieces):
        if start[i] + len(piece) <= body_at:
            reasoning.append(i)
            continue
        if not piece:                      # EOS and friends
            continue
        if ";" in piece:                   # separator belongs to no item
            if cur:
                items.append(cur)
                cur = []
            continue
        cur.append(i)
    if cur:
        items.append(cur)

    out = []
    for steps in items:
        label = "".join(pieces[i] for i in steps).strip()
        if label:
            out.append((label, steps))
    return out, reasoning


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chain-id", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--with-reasoning", action="store_true",
                    help="emit the pre-ANSWER reasoning as one extra segment")
    ap.add_argument("--include-truncated", action="store_true",
                    help="keep turns that hit the token cap. Off by default: "
                         "a truncated reply never reaches its final ANSWER, so "
                         "the last `ANSWER:` in it sits inside the reasoning "
                         "and everything after becomes one enormous bogus "
                         "segment. Truncation is also poison-dependent, so "
                         "keeping these compares arms on unequal footing.")
    args = ap.parse_args()

    analytics, src = _load_analytics()
    print(f"reduction from {src}  (DECAY={analytics.DECAY})")

    run = args.run
    turns = capture.load_turns(run / "turns.jsonl")
    meta = __import__("json").loads((run / "run_meta.json").read_text())
    buf, notes = capture.find_run_buffer(run)
    for n in notes:
        print(f"!! {n}")
    tasks = capture.index_tasks(capture.load_buffer(buf))

    rows, skipped, truncated, unprefixed = [], 0, 0, 0
    for t in turns:
        if args.chain_id and t["chain_id"] != args.chain_id:
            continue
        if args.round and t["round_no"] != args.round:
            continue
        if not args.include_truncated and t.get("finish_reason") == "length":
            truncated += 1
            continue
        rec = tasks.get(t.get("request_id"))
        if rec is None:
            skipped += 1
            continue
        uri = (rec.get("attention_stats") or {}).get("uri") or ""
        parts = uri.rstrip("/").split("/")
        prov = run / "attention" / parts[-2] / parts[-1]
        if not prov.is_file():
            skipped += 1
            continue

        pieces = pieces_of(t)
        items, reasoning = segment_steps(pieces)
        if not items:          # no ANSWER: in the visible reply -> nothing to segment
            unprefixed += 1
            continue
        if args.with_reasoning and reasoning:
            items = [("<reasoning>", reasoning)] + items
        n_tok = t["n_prompt_tokens"]
        for si, (label, steps) in enumerate(items):
            score = analytics.aggregate(prov, steps, n_tok)
            rows.append({
                "chain_id": t["chain_id"], "round_no": t["round_no"],
                "task_type": t["task_type"], "model": meta.get("model", ""),
                "segment_index": si, "segment_text": label,
                "decode_steps": " ".join(map(str, steps)),
                "n_prompt_tokens": n_tok,
                "prompt_token_ids": " ".join(
                    map(str, analytics.np.asarray(
                        __import__("safetensors").safe_open(str(prov), "numpy")
                        .get_tensor("prompt_token_ids")).tolist())),
                "scores": " ".join(f"{v:.6g}" for v in score),
            })

    cols = ["chain_id", "round_no", "task_type", "model", "segment_index",
            "segment_text", "decode_steps", "n_prompt_tokens",
            "prompt_token_ids", "scores"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}"
          + (f"   ({skipped} turns had no usable record)" if skipped else "")
          + (f"   ({truncated} turns dropped: hit the token cap, no final "
             f"ANSWER)" if truncated else "")
          + (f"   ({unprefixed} turns dropped: replied without an ANSWER: "
             f"prefix)" if unprefixed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
