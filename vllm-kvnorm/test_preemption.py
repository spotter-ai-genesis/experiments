# SPDX-License-Identifier: Apache-2.0
"""Preemption must be invisible to the capture.

``Scheduler._preempt_request`` frees a request's blocks directly, bypassing the
connector hook. That sounds alarming, but it is a non-issue:

* preemption resets ``num_computed_tokens`` to 0 and requeues the request, which
  is then **fully recomputed** into fresh blocks;
* our hook only ever fires at true finish, and pins whichever blocks hold the
  *final* KV;
* so the discarded intermediate KV is regenerated, and nothing is missed.

This test forces heavy preemption and checks the capture is unaffected: every
request is captured exactly once, with a token count matching the generation.

Run: python test_preemption.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

PROMPTS = [
    "The capital of France is",
    "Explain how a four stroke engine works, covering intake, compression, power and exhaust.",
    "List the planets in order.",
    "Describe the water cycle: evaporation, condensation, precipitation and runoff.",
    "What is 2+2?",
    "Describe the history of the printing press and its effect on literacy.",
    "Name three primary colours.",
    "Summarise the causes of the fall of the Western Roman Empire.",
]

WORKFLOW_ID = "kvnorm-preemption"


def _child(model: str, blocks: int, max_model_len: int) -> int:
    from flowcept import Flowcept
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    # Count preemptions so we can prove the workload actually exercises them.
    stats = {"preemptions": 0, "hook_calls": 0}
    original_preempt = Scheduler._preempt_request

    def traced_preempt(self, request, *a, **kw):
        stats["preemptions"] += 1
        return original_preempt(self, request, *a, **kw)

    Scheduler._preempt_request = traced_preempt

    from vllm_kvnorm import connector

    original_hook = connector._SchedulerSide.request_finished

    def traced_hook(self, request):
        # The streaming connector needs no block ids here: every token was
        # already scored in the step that wrote it.
        stats["hook_calls"] += 1
        return original_hook(self, request)

    connector._SchedulerSide.request_finished = traced_hook

    with Flowcept("vllm", workflow_id=WORKFLOW_ID, workflow_name="kvnorm_preemption") as fc:
        llm = LLM(
            model=model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="KVNormConnector",
                kv_connector_module_path="vllm_kvnorm",
                kv_role="kv_producer",
                kv_connector_extra_config={"workflow_id": WORKFLOW_ID},
            ),
            max_model_len=max_model_len,
            gpu_memory_utilization=0.70,
            enforce_eager=True,
            enable_prefix_caching=False,
            num_gpu_blocks_override=blocks,
        )
        outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=32))

        # Ground truth: what the engine actually produced per request.
        truth = {
            int(o.request_id.split("-", 1)[0]): len(o.prompt_token_ids)
            + len(o.outputs[0].token_ids)
            for o in outs
        }
        # Drive one more step so the final scores are computed and released.
        llm.generate(["ping"], SamplingParams(temperature=0.0, max_tokens=1))
        del llm

        captured = [
            {
                "request_id": r["used"]["request_id"],
                "num_computed_tokens": r["used"]["num_computed_tokens"],
                "num_scores": len(r["generated"]["score"]),
            }
            for r in fc.get_buffer()
            if r.get("type") == "task" and r.get("used", {}).get("request_id")
        ]

    print("TRUTH " + json.dumps(truth))
    print("CAPTURED " + json.dumps(captured))
    print(f"STATS preemptions={stats['preemptions']} hook_calls={stats['hook_calls']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--max-model-len", type=int, default=128)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return _child(args.model, args.blocks, args.max_model_len)

    import subprocess

    print(f">>> running {len(PROMPTS)} prompts with only {args.blocks} KV blocks "
          f"(forces preemption)")
    rc = subprocess.run(
        [sys.executable, __file__, "--model", args.model, "--blocks", str(args.blocks),
         "--max-model-len", str(args.max_model_len), "--child"],
        capture_output=True, text=True,
    )
    truth_line = next((l for l in rc.stdout.splitlines() if l.startswith("TRUTH ")), None)
    cap_line = next((l for l in rc.stdout.splitlines() if l.startswith("CAPTURED ")), None)
    stats_line = next((l for l in rc.stdout.splitlines() if l.startswith("STATS ")), None)
    if rc.returncode != 0 or truth_line is None or cap_line is None:
        print(rc.stdout[-3000:]); print(rc.stderr[-3000:])
        print(f"FAIL: child exited {rc.returncode}")
        return 1

    truth = {int(k): v for k, v in json.loads(truth_line[len("TRUTH "):]).items()}
    stats = dict(kv.split("=") for kv in stats_line.split()[1:])
    print("   ", stats_line)

    captured: dict[int, list[dict]] = {}
    for r in json.loads(cap_line[len("CAPTURED "):]):
        captured.setdefault(int(r["request_id"].split("-", 1)[0]), []).append(r)

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    print("\n=== preemption is invisible to the capture ===")
    n_preempt = int(stats["preemptions"])
    check(n_preempt > 0, f"workload actually preempted ({n_preempt} preemptions) -- test is meaningful")
    check(
        int(stats["hook_calls"]) == len(PROMPTS) + 1,  # +1 for the trailing "ping"
        f"hook fired once per request, not once per preemption "
        f"({stats['hook_calls']} calls for {len(PROMPTS)} prompts + 1 ping)",
    )
    check(set(captured) >= set(truth), f"every request captured ({len(captured)}/{len(truth)})")
    check(
        all(len(v) == 1 for v in captured.values()),
        "each request captured exactly once (no duplicates from re-runs)",
    )

    # The final sampled token is never fed back through the model, so no KV is
    # ever written for it: num_computed_tokens == prompt + output - 1. What
    # matters is that this holds *uniformly*, including for preempted requests
    # -- a preemption that lost KV would show up as a larger, variable gap.
    deltas = {
        i: truth[i] - captured[i][0]["num_computed_tokens"]
        for i in sorted(truth)
        if i in captured
    }
    check(
        set(deltas.values()) == {1},
        "captured token count is prompt+output-1 for every request, uniformly "
        f"(distinct deltas: {sorted(set(deltas.values()))})",
    )

    uncovered = [
        i
        for i in sorted(truth)
        if i in captured
        and captured[i][0]["num_scores"] != captured[i][0]["num_computed_tokens"]
    ]
    check(
        not uncovered,
        f"one score per computed token, not a truncated prefix (gaps: {uncovered})",
    )

    print(f"\nFAILURES: {len(failures)}")
    for f_ in failures:
        print("  -", f_)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
