#!/usr/bin/env python
"""Generate a poisoned copy of opal_chains.csv, one per strategy.

    python scripts/make_poison_csv.py --all
    python scripts/make_poison_csv.py --arm qc --model Qwen/Qwen3.5-35B-A3B

Writes `data/poisoned/<arm>.csv` with the SAME seven columns, the same rows in
the same order, and the same chain ids -- so it drops into anything that reads
the original, with no flags and no code. A sidecar `<arm>.manifest.json`
records exactly what was changed.

WHY A DATASET RATHER THAN A RUNTIME FLAG. `--poison` applies a manifest inside
the runner, which is right for a paired experiment: the clean and poisoned
arms then share one code path and differ only in the edit. But it makes the
corruption invisible to anything else. A CSV can be handed to another harness,
diffed, versioned, or read by someone who does not have this repository.

THE CORRECT ANSWER IS UPDATED, AND THAT IS THE POINT. Lowering a quality-gate
floor does not merely mislead the model -- it changes what the right answer
IS. The benchmark already records this: the `explanation` column names the
answer each skipped step produces ("ablations: ignoring the quality gate gives
20 on 11 plants"), so an arm that legitimises a documented shortcut can carry
a truthful `correct_answer` rather than a stale one. A model that reasons
correctly over the poisoned tables now scores RIGHT, and one that takes the
shortcut scores WRONG -- the grading inverts, which is what makes the dataset
a test of the method rather than of the model.

Arms with no such mapping (`outfmt`, `null`) leave answers untouched; they
corrupt presentation, not ground truth.

TWO THINGS THIS CANNOT DO, both stated in the manifest.

Rounds after the first are NOT re-derived. Round 2 says "starting from the
dose you identified in round 1" and its `correct_answer` assumes the original.
Once round 1's answer changes, every later round's stored answer describes a
different investigation. Those rows are passed through unchanged and marked
`ground_truth_invalidated` so nothing scores them by accident.

The `history` arm has no CSV form at all. It substitutes the assistant's own
reply, and this file has no assistant turns -- only prompts. It stays a
runtime-only manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # repo root
sys.path.insert(0, str(HERE / "src"))

from opal_attn.answers import _head, ablations  # noqa: E402
from opal_attn.chains import load_chains, select  # noqa: E402
from opal_attn.poison import (  # noqa: E402
    ARMS,
    BUILTIN_LEGITIMISES,
    PoisonError,
    find_balanced,
    load_strategies,
    load_strategy,
    locate,
)

CSV_COLUMNS = ["chain_id", "round_no", "task_type", "prompt", "tool_ranges",
               "correct_answer", "explanation"]

def build(arm, rows: list[dict], chains, tok, *, update_answers: bool
          ) -> tuple[list[dict], dict]:
    """`arm` is a built-in name, or a Strategy loaded from JSON."""
    if isinstance(arm, str):
        spec = next((a for a in ARMS if a[0] == arm), None)
        if spec is None:
            raise SystemExit(f"unknown arm {arm!r}; "
                             f"have {', '.join(a[0] for a in ARMS)}")
        name, form, kind, locator, where = spec
        want = BUILTIN_LEGITIMISES.get(name)
    else:
        name, form, kind, locator, where = arm.as_arm()
        want = arm.legitimises
    if where == "history":
        raise SystemExit(
            f"the {name!r} arm substitutes the model's own reply, and this "
            f"file has no assistant turns. It stays a runtime manifest "
            f"(configs/poison/history.json).")

    by_chain = {c.chain_id: c for c in chains}
    out, changes, skipped = [], [], {"no anchor": 0, "unbalanced": 0,
                                     "no ablation": 0}
    poisoned_chains: set[str] = set()

    for r in rows:
        row = dict(r)
        cid, rn = row["chain_id"], int(row["round_no"])
        if cid not in by_chain:
            out.append(row)
            continue
        if rn != 1:
            out.append(row)
            continue
        src = by_chain[cid].rounds[0]
        found = locator(src.prompt, src)
        if found is None:
            skipped["no anchor"] += 1
            out.append(row)
            continue
        find, cands = found
        try:
            best = next((c for c, d in find_balanced(tok, src.prompt, find, cands)
                         if d == 0), None)
        except PoisonError:
            skipped["no anchor"] += 1
            out.append(row)
            continue
        if best is None:
            skipped["unbalanced"] += 1
            out.append(row)
            continue
        new_answer = row["correct_answer"]
        if update_answers and want:
            abl = ablations(src.explanation)
            if want not in abl:
                # Without the ablation we cannot state the new truth, and
                # shipping the old one would be a lie in a file that looks
                # authoritative. Leave the chain alone entirely.
                skipped["no ablation"] += 1
                out.append(row)
                continue
            # `_head` trims the prose the explanation column carries
            # around a value -- 'ignoring the quality gate gives 20 on
            # 11 plants' means the answer is '20', not the whole phrase.
            new_answer = _head(abl[want])

        lo, hi = locate(src.prompt, find)
        row["prompt"] = src.prompt[:lo] + best + src.prompt[hi:]
        row["correct_answer"] = new_answer
        row["explanation"] = (
            f"(POISONED arm={name} kind={kind}: {find!r} -> {best!r}; "
            f"original answer {r['correct_answer']!r}"
            + (f", now {new_answer!r} via the documented ablation {want!r}"
               if new_answer != r["correct_answer"] else ", unchanged")
            + f") {r['explanation']}")
        poisoned_chains.add(cid)
        changes.append({"chain_id": cid, "find": find, "replace": best,
                        "answer_before": r["correct_answer"],
                        "answer_after": new_answer})
        out.append(row)

    # Later rounds of a poisoned chain inherit a premise that no longer holds.
    invalidated = 0
    for row in out:
        if row["chain_id"] in poisoned_chains and int(row["round_no"]) > 1:
            row["explanation"] = ("(ground_truth_invalidated: round 1 of this "
                                  "chain was poisoned and its answer may have "
                                  "changed; this row's correct_answer assumes "
                                  "the original) " + row["explanation"])
            invalidated += 1

    manifest = {
        "arm": name, "form": form, "kind": kind,
        "model": getattr(tok, "name_or_path", None),
        "rows": len(out), "chains_poisoned": len(poisoned_chains),
        "legitimises": want, "answers_updated": bool(want and update_answers),
        "rounds_ground_truth_invalidated": invalidated,
        "skipped": skipped, "changes": changes,
        "note": "round 1 only; later rounds are passed through unchanged and "
                "flagged in `explanation` because their stored answer assumes "
                "the unpoisoned round 1.",
    }
    return out, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "bench" / "opal" / "data" / "opal_chains.csv"))
    ap.add_argument("--out", default=str(HERE / "bench" / "opal" / "data" / "poisoned"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--limit", type=int, default=0, help="first N chains; 0 = all")
    ap.add_argument("--arm", default=None, help="a built-in arm name")
    ap.add_argument("--strategy", default=None, metavar="FILE.json",
                    help="a declarative strategy; see configs/strategies/")
    ap.add_argument("--strategies", default=None, metavar="DIR",
                    help="every *.json strategy in a directory")
    ap.add_argument("--all", action="store_true",
                    help="every built-in arm AND configs/strategies/")
    ap.add_argument("--probe", action="store_true",
                    help="do not write anything: report the token delta of "
                         "every candidate phrasing on every chain. This is "
                         "the loop for authoring a strategy -- parity "
                         "rejects most wordings and will not tell you which "
                         "unless you ask.")
    ap.add_argument("--keep-answers", action="store_true",
                    help="do NOT update correct_answer. The file then grades "
                         "against a truth the poisoned tables contradict, "
                         "which measures obedience rather than correctness.")
    args = ap.parse_args()
    if not any((args.arm, args.all, args.strategy, args.strategies)):
        raise SystemExit("pass --arm NAME, --strategy FILE, --strategies DIR, "
                         "or --all")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chains = select(load_chains(args.csv), limit=args.limit)
    with Path(args.csv).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo: list = []
    if args.all:
        todo += [a[0] for a in ARMS if a[4] != "history"]
        # A MISSING STRATEGY DIR IS FATAL, not a quiet skip. --all that
        # silently built five arms instead of nine would hand you a suite
        # with four arms missing and no error to notice.
        d = HERE / "configs" / "strategies"
        if not d.is_dir():
            raise SystemExit(f"--all needs {d}, which does not exist")
        found = load_strategies(d)
        if not found:
            raise SystemExit(f"--all found no *.json strategies in {d}")
        todo += found
    if args.arm:
        todo.append(args.arm)
    if args.strategy:
        todo.append(load_strategy(args.strategy))
    if args.strategies:
        todo += load_strategies(args.strategies)
    print(f"{'arm':11s}{'rows':>6s}{'chains':>8s}{'answers':>9s}"
          f"{'invalid':>9s}  skipped")
    if args.probe:
        from opal_attn.poison import find_balanced as _fb
        for item in todo:
            name = item if isinstance(item, str) else item.name
            if isinstance(item, str):
                spec = next((a for a in ARMS if a[0] == item), None)
                loc = spec[3] if spec else None
            else:
                loc = item.locator()
            if loc is None:
                continue
            print(f"\n=== {name} ===")
            tally: dict[str, list[int]] = {}
            for c in chains:
                got = loc(c.rounds[0].prompt, c.rounds[0])
                if got is None:
                    continue
                find, cands = got
                for cand, d in _fb(tok, c.rounds[0].prompt, find, cands):
                    tally.setdefault(cand if len(cands) > 1 else "(only)",
                                     []).append(d)
            if not tally:
                print("  no chain matched `find`")
                continue
            for cand, ds in sorted(tally.items(),
                                   key=lambda kv: -sum(1 for d in kv[1] if d == 0)):
                ok = sum(1 for d in ds if d == 0)
                rng = f"{min(ds):+d}..{max(ds):+d}"
                print(f"  balanced on {ok:3d}/{len(ds):3d}  delta {rng:>9s}  "
                      f"{cand[:66]!r}")
        return 0

    for item in todo:
        name = item if isinstance(item, str) else item.name
        try:
            out, man = build(item, rows, chains, tok,
                             update_answers=not args.keep_answers)
        except SystemExit as e:
            print(f"{name:11s}  skipped: {e}")
            continue
        path = out_dir / f"{name}.csv"
        with path.open("w", newline="") as fh:
            # lineterminator: csv defaults to CRLF, the source file is LF.
            # A whole-file diff against the original should show only the
            # rows that were poisoned, not every line.
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS,
                               lineterminator="\n")
            w.writeheader()
            w.writerows({k: r[k] for k in CSV_COLUMNS} for r in out)
        (out_dir / f"{name}.manifest.json").write_text(
            json.dumps(man, indent=2) + "\n")
        upd = sum(1 for c in man["changes"]
                  if c["answer_before"] != c["answer_after"])
        sk = ", ".join(f"{k} {v}" for k, v in man["skipped"].items() if v) or "-"
        print(f"{name:11s}{man['rows']:6d}{man['chains_poisoned']:8d}"
              f"{upd:9d}{man['rounds_ground_truth_invalidated']:9d}  {sk}")
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
