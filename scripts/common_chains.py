#!/usr/bin/env python
"""The chains every poisoning arm actually poisons.

    python scripts/common_chains.py                  # one id per line
    python scripts/common_chains.py --oneline        # space separated
    python scripts/common_chains.py --report         # the full coverage table

An arm poisons a chain only where its anchor exists AND the substitution is
token-balanced, so the arms cover different subsets: 20 chains for `outfmt`
and `null`, 13 for the rest -- and not the same 13.

WHY THAT MATTERS. Compared across all 20, a chain an arm never touched scores
as "the poison did nothing", which is indistinguishable from "the poison was
applied and the model resisted". That biases every arm toward looking weaker,
and unevenly, because the arms miss different chains. Pinning every arm to the
intersection makes the comparison apples-to-apples: every chain in the run is
poisoned in every arm.

The intersection is ALWAYS taken over every manifest present, never over the
arms you happen to be running. Otherwise the pinned set would change with the
arm list and two invocations would stop being comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POISONED = ROOT / "bench" / "opal" / "data" / "poisoned"


def arm_sets(d: Path = POISONED) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for m in sorted(d.glob("*.manifest.json")):
        j = json.loads(m.read_text())
        out[j["arm"]] = {c["chain_id"] for c in j.get("changes", [])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=POISONED)
    ap.add_argument("--oneline", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    sets = arm_sets(args.dir)
    if not sets:
        sys.exit(f"error: no *.manifest.json in {args.dir}. "
                 f"Build the datasets first: scripts/make_poison_csv.py --all")
    core = sorted(set.intersection(*sets.values()))

    if args.report:
        union = set().union(*sets.values())
        print(f"{'arm':12} {'poisons':>8} {'in core':>8} {'outside':>8}")
        for a, s in sorted(sets.items()):
            print(f"  {a:12} {len(s):>6} {len(s & set(core)):>8} {len(s - set(core)):>8}")
        print(f"\nunion poisoned by some arm : {len(union)}")
        print(f"poisoned by EVERY arm      : {len(core)}")
        if union - set(core):
            print("\nnot poisoned by every arm:")
            for c in sorted(union - set(core)):
                miss = [a for a, s in sorted(sets.items()) if c not in s]
                print(f"  {c:26} missing from: {', '.join(miss)}")
        return 0

    if not core:
        sys.exit(
            "error: the arms share no poisoned chain, so there is nothing to\n"
            "pin them to. The candidate pool is too small: an arm poisons only\n"
            "where its anchor exists AND the substitution is token-balanced.\n"
            "\n"
            "Measured on this benchmark (OPAL_LIMIT -> common core):\n"
            "    3 -> 0    5 -> 1    10 -> 4    15 -> 7    20 -> 10    30 -> 14\n"
            "\n"
            "Either widen the pool:        OPAL_LIMIT=20 ./scripts/run_suite.sh\n"
            "or run without the pin:       OPAL_PIN=0 ./scripts/run_suite.sh\n"
            "The second is fine for a smoke test, where you are checking that\n"
            "the capture works, not comparing arms.")
    print(" ".join(core) if args.oneline else "\n".join(core))
    return 0


if __name__ == "__main__":
    sys.exit(main())
