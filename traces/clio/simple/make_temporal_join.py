#!/usr/bin/env python3
"""Temporal-join demonstration for the SPOTTER fused clio+kvnorm dataset.

The two provenance streams share no direct key yet (Stage 3 will stamp the vLLM
response id onto clio's LM record). This script demonstrates the correlation
that IS recoverable today: both streams are 1:1, sequential, and timestamped, so
each clio ai_model_invocation aligns by time-order to exactly one kvnorm
kv_token_importance record. Each clio call is resolved back to the agent
prompt/session/turn it belongs to, so the join reads as
"agent message  <->  per-token KV-importance array".

Inputs : tasks_<job>.json, workflows_<job>.json (mongoexport NDJSON)
Outputs: temporal_join.md (human table) + temporal_join.json (machine-readable)
"""
import json, sys, glob, statistics

DSET = "/work/nvme/bekn/jcernuda/spotter/spotter_dataset_3082983"
PROMPTS = [
    "What is 17 multiplied by 23? Show your reasoning step by step.",
    "Explain the difference between a stack and a queue data structure in one paragraph.",
    "What are the three laws of thermodynamics? Give a one-sentence summary of each.",
    "If a train travels at 60 mph for 2.5 hours, how far does it go? Show the calculation.",
    "Write a haiku about machine learning.",
]

def dt(v):
    return v.get("$date") if isinstance(v, dict) else v

def num(v):
    if isinstance(v, dict):
        return v.get("$numberLong") or v.get("$numberInt") or v.get("$numberDouble")
    return v

def find_scores(d):
    """Return the per-token score list in a kvnorm record (a list of floats)."""
    best = None
    def rec(x):
        nonlocal best
        if isinstance(x, list) and x and all(isinstance(e, (int, float)) for e in x):
            if best is None or len(x) > len(best):
                best = x
        elif isinstance(x, dict):
            for v in x.values(): rec(v)
        elif isinstance(x, list):
            for v in x: rec(v)
    rec(d)
    return best

def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out

tasks = load(glob.glob(f"{DSET}/tasks_*.json")[0])
wfs   = load(glob.glob(f"{DSET}/workflows_*.json")[0])

kv, llm = [], []
for d in tasks:
    u = d.get("used") or {}
    if isinstance(u, dict) and "chatcmpl" in str(u.get("request_id")):
        kv.append(d)
    elif d.get("subtype") == "ai_model_invocation":
        llm.append(d)

# workflow_id -> (session_id, turn_status)
wf_map = {}
for w in wfs:
    cm = (w.get("custom_metadata") or {}).get("clio") or {}
    sid = cm.get("session_id")
    if sid:
        wf_map[w["workflow_id"]] = (sid, cm.get("state_event_type", "?"))

kv.sort(key=lambda d: str(dt(d.get("started_at"))))
llm.sort(key=lambda d: str(dt(d.get("started_at"))))

# session_id -> prompt index, in first-appearance (time) order
session_order = []
for d in llm:
    sid = wf_map.get(d.get("workflow_id"), (d.get("workflow_id"), "?"))[0]
    if sid not in session_order:
        session_order.append(sid)
sid_to_prompt = {sid: PROMPTS[i] if i < len(PROMPTS) else "(unknown)"
                 for i, sid in enumerate(session_order)}

assert len(kv) == len(llm), f"stream count mismatch {len(kv)} vs {len(llm)}"

pairs = []
for i, (c, k) in enumerate(zip(llm, kv), start=1):
    sid, status = wf_map.get(c.get("workflow_id"), (c.get("workflow_id"), "?"))
    ku = k.get("used") or {}
    scores = find_scores(k) or []
    n_tok = num(ku.get("num_computed_tokens"))
    pairs.append({
        "pair": i,
        "prompt": sid_to_prompt.get(sid, "(unknown)"),
        "session_id": sid,
        "turn_status": status,
        "clio": {
            "workflow_id": c.get("workflow_id"),
            "task_id": c.get("task_id"),
            "model_id": (((c.get("custom_metadata") or {}).get("clio") or {}).get("provider") or {}).get("model_id"),
            "logged_at": dt(c.get("started_at")),
        },
        "kvnorm": {
            "request_id": ku.get("request_id"),
            "num_computed_tokens": n_tok,
            "score_len": len(scores),
            "score_min": round(min(scores), 5) if scores else None,
            "score_max": round(max(scores), 5) if scores else None,
            "score_mean": round(statistics.mean(scores), 5) if scores else None,
            "metric": ku.get("metric") or k.get("used", {}).get("metric"),
            "started_at": dt(k.get("started_at")),
            "ended_at": dt(k.get("ended_at")),
        },
    })

with open(f"{DSET}/temporal_join.json", "w") as f:
    json.dump({"note": "temporal (time-order) join; no shared key yet — Stage 3 will add vLLM response id",
               "n_pairs": len(pairs), "pairs": pairs}, f, indent=2)

# ---- markdown ----
lines = []
lines.append("# Temporal-join demonstration — clio agent messages ↔ kvnorm KV-importance\n")
lines.append("Fused dataset job **3082983** (Granite-4.2-30b on GH200). Both provenance streams")
lines.append("landed in one Flowcept MongoDB. They share **no direct key yet** — this alignment is")
lines.append("by **time-order** (both streams are 1:1, sequential, timestamped). Stage 3 will stamp")
lines.append("the vLLM response id onto the clio LM record for a bulletproof key.\n")
lines.append(f"**{len(pairs)} LM calls ↔ {len(pairs)} kvnorm records**, across "
             f"{len(session_order)} agent sessions (prompts).\n")
lines.append("| # | prompt (agent turn) | turn | clio LM logged_at | kvnorm request_id | KV tokens | score len | score mean (min→max) |")
lines.append("|--|--|--|--|--|--|--|--|")
for p in pairs:
    k = p["kvnorm"]
    rng = f"{k['score_mean']} ({k['score_min']}→{k['score_max']})" if k["score_mean"] is not None else "—"
    lines.append(f"| {p['pair']} | {p['prompt'][:46]} | {p['turn_status'].replace('turn.','')} "
                 f"| {p['clio']['logged_at']} | `{k['request_id']}` | {k['num_computed_tokens']} "
                 f"| {k['score_len']} | {rng} |")
lines.append("\n## How to read it")
lines.append("- Each row is one LM call the agent made: the **prompt/session** it served (clio side)")
lines.append("  aligned to the **per-token KV-importance array** vLLM+kvnorm produced for it.")
lines.append("- `KV tokens` = `num_computed_tokens` (prompt+generated); `score len` matches it —")
lines.append("  one importance float per token (mean over 64 Granite layers & KV heads,")
lines.append("  metric = PagedEviction ‖V‖₂/‖K‖₂).")
lines.append("- Multi-row sessions (e.g. prompt 2) are **multi-step ReAct loops** — the KV cache")
lines.append("  grows across steps, so later calls score longer token arrays.")
lines.append("\n## Caveat (why Stage 3)")
lines.append("Time-order pairing is correct here only because requests ran strictly sequentially")
lines.append("(vLLM `Running: 1 req`). Under concurrency it would be ambiguous. Recording the vLLM")
lines.append("`chatcmpl-*` response id on clio's `ai_model_invocation` gives a direct, robust join.")

with open(f"{DSET}/temporal_join.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\n[written] {DSET}/temporal_join.md  +  temporal_join.json")
