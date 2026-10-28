#!/usr/bin/env python
"""Build one poison manifest per arm, with token parity enforced per chain.

    python poison/build_runtime_manifest.py --limit 20 --model Qwen/Qwen3.5-35B-A3B

Writes configs/poison/<arm>-runtime.json and prints a coverage table.

ONLY NEEDED FOR `history`. Every other arm is a dataset built by
make_poison_csv.py; a runtime manifest exists because the history arm
substitutes the assistant's own reply, which a CSV cannot express. Rebuild it
whenever you change model -- token parity is tokenizer-dependent.

WHY A BUILD STEP. The text to poison differs per chain, so an arm is not one
edit but twenty, each of which must (a) match exactly once and (b) preserve the
token count under the run's own tokenizer. Both are checked here, with no GPU,
so an unbalanced substitution is a build error rather than a silently re-based
distance axis discovered after a capture.

COVERAGE IS NOT UNIFORM AND THAT IS THE DATA, NOT A BUG. Seven of the first
twenty chains have no quality gate in round 1 at all -- one rule, one tool --
so the qc, rules and schema arms cannot apply to them. Each arm therefore
records the chains it covers, and cross-arm comparisons are valid only on the
intersection. Per-arm clean-vs-poisoned pairing is unaffected, since that
pairs a chain with itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # repo root
sys.path.insert(0, str(HERE / "src"))

from opal_attn.chains import load_chains, select  # noqa: E402
from opal_attn.poison import (  # noqa: E402
    ARMS,
    PoisonError,
    find_balanced,
    load_strategies,
    locate,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "bench" / "opal" / "data" / "opal_chains.csv"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "configs" / "poison"))
    ap.add_argument("--strategies", default=str(HERE / "configs" / "strategies"),
                    help="directory of declarative strategies to include "
                         "alongside the built-in arms. They need runtime "
                         "manifests too, or a strategy can be generated as a "
                         "CSV and not run through --poison.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chains = select(load_chains(args.csv), limit=args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_arms = list(ARMS)
    sdir = Path(args.strategies)
    if sdir.is_dir():
        all_arms += [st.as_arm() for st in load_strategies(sdir)]

    coverage: dict[str, set[str]] = {}
    print(f"{'arm':12s}{'form':12s}{'covered':>9s}{'no anchor':>11s}{'unbalanced':>12s}")
    for name, form, kind, locator, where in all_arms:
        spec = {"arm": name, "form": form, "kind": kind,
                "model": args.model, "round_no": 1, "where": where, "chains": {}}
        noanchor = unbalanced = 0
        for c in chains:
            row = c.rounds[0]
            found = locator(row.prompt, row)
            if found is None:
                noanchor += 1
                continue
            find, cands = found
            if where == "history":
                spec["chains"][c.chain_id] = {"replace": cands[0]}
                continue
            try:
                ranked = find_balanced(tok, row.prompt, find, cands)
            except PoisonError:
                noanchor += 1
                continue
            best = next((c_ for c_, d in ranked if d == 0), None)
            if best is None:
                unbalanced += 1
                continue
            lo, hi = locate(row.prompt, find)
            spec["chains"][c.chain_id] = {"find": find, "replace": best,
                                          "char_lo": lo, "char_hi": hi}
        (out / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")
        coverage[name] = set(spec["chains"])
        print(f"{name:12s}{form:12s}{len(spec['chains']):9d}{noanchor:11d}{unbalanced:12d}")

    common = set.intersection(*coverage.values()) if coverage else set()
    print(f"\ncommon chain set (all arms apply): {len(common)} of {len(chains)}")
    print("cross-arm comparisons are valid only on that intersection; "
          "per-arm clean-vs-poisoned pairing uses each arm's own coverage.")
    (out / "coverage.json").write_text(json.dumps(
        {"per_arm": {k: sorted(v) for k, v in coverage.items()},
         "common": sorted(common)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
