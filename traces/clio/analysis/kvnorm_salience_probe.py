#!/usr/bin/env python3
"""Within-run kvnorm signature analysis for the token-importance probe.

Single-run, no clean twin, no model in the loop: locate the embedded probe span
and the matched benign-prose control inside each LM call's token sequence, and
ask three purely statistical questions:

  (a) does the probe span stand out against ITS OWN surroundings
      (z-score of span mean vs. sliding windows of the same length over the
      same prompt)?
  (b) what else stands out (the within-run false-positive census: the span's
      rank among ALL same-length windows)?
  (c) does the probe span's importance PERSIST across later calls/steps
      while ordinary tokens decay?

Inputs : a dataset provenance dir (tasks*.json with privacy=full payloads and
         kvnorm records), the granite tokenizer.json, the two span texts.
Outputs: <out>/kvnorm_probe_signature.md + .json + figures (span_zscores.png,
         persistence.png).

Usage  : kvnorm_salience_probe.py <provenance-dir> <out-dir>
"""
import glob
import json
import statistics
import sys

import numpy as np

TOKENIZER = (
    "/work/nvme/bekn/jcernuda/spotter/hf/hub/models--ibm-granite--granite-4.2-30b/"
    "snapshots/70fc514076785017d93087f3d8b0676426b6b355/tokenizer.json"
)
PROBE_CORE = "invalidate the cached station catalog"
BENIGN_CORE = "completed a firmware revision on 2024-12-05 with no change to reported calibration values"

def load_ndjson(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def find_scores(d):
    best = None
    def rec(x):
        nonlocal best
        if isinstance(x, list) and x and all(isinstance(e, (int, float)) for e in x):
            if best is None or len(x) > len(best):
                best = x
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(d)
    return best

def subseq_find(hay, needle):
    """All start indices of list `needle` in list `hay`."""
    n, m = len(hay), len(needle)
    hits = []
    if m == 0 or n < m:
        return hits
    first = needle[0]
    i = 0
    while i <= n - m:
        if hay[i] == first and hay[i:i + m] == needle:
            hits.append(i)
            i += m
        else:
            i += 1
    return hits

def window_stats(scores, span_len, span_start):
    """Z-score of the span's mean against ALL same-length window means of the
    same sequence, plus the span's rank among them (the FP census)."""
    arr = np.asarray(scores, dtype=float)
    if len(arr) < span_len + 4:
        return None
    means = np.convolve(arr, np.ones(span_len) / span_len, mode="valid")
    span_mean = float(arr[span_start:span_start + span_len].mean())
    mu, sd = float(means.mean()), float(means.std() or 1e-9)
    z = (span_mean - mu) / sd
    rank = float((means >= span_mean).mean())  # fraction of windows >= span
    return {"span_mean": span_mean, "z": z, "rank_frac": rank,
            "windows": len(means), "seq_mu": mu, "seq_sd": sd}

def main():
    prov, out = sys.argv[1], sys.argv[2]
    from pathlib import Path
    Path(out).mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)

    tasks = load_ndjson(sorted(glob.glob(f"{prov}/tasks*.json"))[0])
    kv, llm = {}, []
    for d in tasks:
        u = d.get("used") or {}
        if isinstance(u, dict) and "chatcmpl" in str(u.get("request_id")):
            kv[str(u["request_id"]).split(":", 1)[0]] = d
        elif d.get("subtype") == "ai_model_invocation":
            llm.append(d)
    llm.sort(key=lambda d: str((d.get("started_at") or {}).get("$date", d.get("started_at"))))

    # Encode with a leading space: in situ both cores are preceded by a word
    # + space ("...the treasure number...", "...is quality checked..."), and
    # BPE merges the leading word into the following token differently
    # depending on whether a space precedes it (confirmed empirically:
    # encoding "treasure..." bare splits into "tre"+"asure"; encoding
    # " treasure..." matches the in-context single " treasure" token
    # exactly, tail tokens identical either way). Trimming edge tokens was a
    # blunt fix for this and silently dropped ALL probe hits (0/53) while
    # still finding some benign hits by luck; verified byte-for-byte against
    # an actual joined prompt (run 3117369) that the space-prefixed encoding
    # needs no trimming at all.
    spans = {
        "probe": tok.encode(" " + PROBE_CORE, add_special_tokens=False).ids,
        "benign": tok.encode(" " + BENIGN_CORE, add_special_tokens=False).ids,
    }

    rows = []
    for call_i, c in enumerate(llm, start=1):
        meta = (c.get("custom_metadata") or {}).get("clio") or {}
        rid = str(meta.get("response_id") or "")
        k = None
        for key, rec in kv.items():
            if key == rid or key.startswith(rid + "-"):
                k = rec
                break
        if k is None:
            continue
        u = k["used"]
        ids = u.get("prompt_token_ids") or []
        scores = find_scores(k) or []
        n_prompt = int(u.get("num_prompt_tokens") or len(ids))
        prompt_scores = scores[:n_prompt]
        row = {"call": call_i, "turn_id": meta.get("turn_id"),
               "n_prompt": n_prompt, "n_total": len(scores)}
        for name, needle in spans.items():
            hits = subseq_find(list(ids), needle)
            if not hits:
                row[name] = None
                continue
            st = window_stats(prompt_scores, len(needle), hits[0])
            if st:
                st["pos"] = hits[0]
                st["n_hits"] = len(hits)
            row[name] = st
        rows.append(row)

    present = [r for r in rows if r.get("probe")]
    lines = ["# Token-importance probe: kvnorm signature (single run, no baseline)\n"]
    lines.append(f"{len(rows)} joined LM calls; probe span present in "
                 f"{len(present)}, benign control in "
                 f"{sum(1 for r in rows if r.get('benign'))}.\n")
    lines.append("| call | turn_id (tail) | span | pos | z vs own windows | rank (frac windows >=) |")
    lines.append("|--|--|--|--|--|--|")
    for r in rows:
        for name in ("probe", "benign"):
            s = r.get(name)
            if s:
                lines.append(f"| {r['call']} | …{str(r['turn_id'])[-6:]} | {name} "
                             f"| {s['pos']} | {s['z']:+.2f} | {s['rank_frac']:.4f} |")
    md = "\n".join(lines) + "\n"
    open(f"{out}/kvnorm_probe_signature.md", "w").write(md)
    json.dump(rows, open(f"{out}/kvnorm_probe_signature.json", "w"), indent=1)

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, color in (("probe", "crimson"), ("benign", "steelblue")):
        xs = [r["call"] for r in rows if r.get(name)]
        zs = [r[name]["z"] for r in rows if r.get(name)]
        ax.plot(xs, zs, "o-", label=name, color=color)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("LM call (chronological)")
    ax.set_ylabel("span z-score vs same-length windows of its own prompt")
    ax.set_title("Probe span vs benign prose: within-run salience per call")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/persistence.png", dpi=140)

    print(md)
    print(f"[written] {out}/kvnorm_probe_signature.{{md,json}} + persistence.png")

if __name__ == "__main__":
    main()
