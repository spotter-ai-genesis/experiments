#!/usr/bin/env python3
"""
Per-reply-segment prefill attention for one opal chain.

A round's answer is a list of items separated by ";" -- "NIL 373; DIVERGES" is
two. Each item is produced by its own run of decode steps, and each of those
steps attended over the whole prompt. This collapses that to one vector per
item: for prefill token j,

    score[j] = sum_i  0.5^i * val_all_max[step_i][j]

over the item's content steps, i counted from the item's first step. The decay
weights the step that commits to the item above the ones that merely spell it
out. Positions outside the connector's top-`top_pct` selection score 0.

The "ANSWER:" prefix tokens and the ";" separators are not part of any item and
are excluded.

Input is a chain CSV plus a chain id. History is the model's own answers by
default; --ground-truth-history teacher-forces it instead, which is what you
want when studying a single round's evidence rather than error propagation.

Output is a CSV, one row per (round, segment):

    chain_id, round_no, task_type, model, segment_index, segment_text,
    decode_steps, n_prompt_tokens, prompt_token_ids, scores

`scores` is a space-separated float per prefill token, same length and order as
the round's prompt. `prompt_token_ids` carries that prompt, so the CSV is
self-contained: render_heatmap.py needs nothing but this file.

Usage
-----
    attn_analytics.py ../data/opal_chains.csv --chain-id exp32-chain-0020-hard \\
        --model cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit \\
        --ground-truth-history --out attn_qwen.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

def _find_connector() -> Path:
    """Locate vllm-attn-connector by walking up, not by a fixed depth."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "vllm-attn-connector"
        if (cand / "src" / "vllm_attn_connector").is_dir():
            return cand
    sys.exit("error: vllm-attn-connector not found above this file")


CONNECTOR = _find_connector()
sys.path.insert(0, str(CONNECTOR / "src"))
sys.path.insert(0, str(CONNECTOR / "tests"))

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

DECAY = 0.5
ANSWER_PREFIX_RE = re.compile(r"^\s*ANSWER\s*:\s*", re.IGNORECASE)


def load_chain(path: Path, chain_id: str) -> list[dict]:
    csv.field_size_limit(1 << 30)
    rows = [r for r in csv.DictReader(path.open()) if r["chain_id"] == chain_id]
    if not rows:
        sys.exit(f"error: no chain {chain_id!r} in {path}")
    return sorted(rows, key=lambda r: int(r["round_no"]))


def segment_steps(out_ids: list[int], tok) -> list[tuple[str, list[int]]]:
    """Split a generated answer into ';'-separated items and their steps.

    Works in decode-step space: step i produced out_ids[i]. The "ANSWER:"
    prefix and the ";" separators belong to no item.
    """
    pieces = [tok.decode([t]) for t in out_ids]
    text = "".join(pieces)
    body_at = len(ANSWER_PREFIX_RE.match(text).group(0)) if ANSWER_PREFIX_RE.match(text) else 0

    # Whitespace inside an item is content: "NIL 373" is one item spanning the
    # space, and dropping it would renumber the decay weights.
    special = set(tok.all_special_ids)
    start, items, cur, char = [], [], [], 0
    for i, piece in enumerate(pieces):
        start.append(char)
        char += len(piece)
    for i, piece in enumerate(pieces):
        if start[i] + len(piece) <= body_at:           # inside "ANSWER:"
            continue
        if out_ids[i] in special:          # EOS and friends end the answer
            continue
        if ";" in piece:                               # separator ends an item
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
    return out


def aggregate(prov: Path, steps: list[int], n_tokens: int) -> np.ndarray:
    with safe_open(str(prov), "numpy") as g:
        pos, vmax = g.get_tensor("topk_pos"), g.get_tensor("val_all_max")
    score = np.zeros(n_tokens, dtype=np.float64)
    for i, s in enumerate(steps):
        if s >= len(pos):
            continue
        w = DECAY ** i
        p, v = pos[s], vmax[s]
        keep = p >= 0
        np.add.at(score, p[keep], w * v[keep])
    return score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--chain-id", required=True)
    ap.add_argument("--model", default="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
    ap.add_argument("--ground-truth-history", action="store_true",
                    help="teacher-force the history (default: the model's own "
                         "answers carry over)")
    ap.add_argument("--out", type=Path, default=Path("attn_analytics.csv"))
    ap.add_argument("--prov-dir", type=Path, default=Path("./attn_prov"))
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--top-pct", type=float, default=10.0)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    import vllm_attn_connector
    assert vllm_attn_connector.install_probe(), "probe failed to install"
    from e2e_example import INSTRUCTION, declare_ranges, parse_answer
    from flowcept import Flowcept

    rounds = load_chain(args.csv, args.chain_id)
    tok = AutoTokenizer.from_pretrained(args.model)
    wf = f"attn-analytics-{args.chain_id}"

    kv = KVTransferConfig(
        kv_connector="AttnConnector", kv_connector_module_path="vllm_attn_connector",
        kv_role="kv_producer",
        kv_connector_extra_config={"workflow_id": wf, "top_pct": args.top_pct,
                                   "out_dir": str(args.prov_dir)})

    captured: list[dict] = []
    with Flowcept("vllm", workflow_id=wf, workflow_name="attn_analytics"):
        llm = LLM(model=args.model, kv_transfer_config=kv,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  **({"max_num_batched_tokens": args.max_num_batched_tokens}
                     if args.max_num_batched_tokens else {}),
                  enforce_eager=True)
        msgs = [{"role": "system", "content": INSTRUCTION}]
        for row in rounds:
            msgs.append({"role": "user", "content": row["prompt"]})
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) + args.max_tokens + 16 > args.max_model_len:
                print(f"  round {row['round_no']}: skipped, {len(ids)} tokens "
                      f"exceeds the window")
                msgs.pop()
                break
            ranges = declare_ranges(text, tok)
            sp = SamplingParams(
                temperature=0.0, max_tokens=args.max_tokens,
                extra_args={"kv_transfer_params":
                            {"ranges": [[d["lo"], d["hi"]] for d in ranges]}})
            out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp,
                               use_tqdm=False)[0].outputs[0]
            got = parse_answer(out.text)
            captured.append({"row": row, "ids": ids, "out_ids": list(out.token_ids),
                             "got": got, "ranges": ranges})
            print(f"  round {row['round_no']} {row['task_type']:<24}"
                  f"got={got!r}  [{len(ids)} tok, {len(out.token_ids)} steps]")
            msgs.append({"role": "assistant",
                         "content": f"ANSWER: {row['correct_answer']}"
                         if args.ground_truth_history else out.text.strip()})
        llm.generate(["."], SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
        del llm

    files = sorted(args.prov_dir.glob("*/*.safetensors"),
                   key=lambda f: int(f.name.split("-")[0]))
    rows_out = []
    for c, prov in zip(captured, files):
        items = segment_steps(c["out_ids"], tok)
        for si, (label, steps) in enumerate(items):
            sc = aggregate(prov, steps, len(c["ids"]))
            rows_out.append({
                "chain_id": args.chain_id, "round_no": c["row"]["round_no"],
                "task_type": c["row"]["task_type"], "model": args.model,
                "segment_index": si, "segment_text": label,
                "decode_steps": ",".join(map(str, steps)),
                "n_prompt_tokens": len(c["ids"]),
                "prompt_token_ids": " ".join(map(str, c["ids"])),
                "scores": " ".join(f"{x:.6g}" for x in sc)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {len(rows_out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
