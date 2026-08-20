# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test: run vLLM with the KV-norm connector, validate the
Flowcept provenance it produces.

Boots a real vLLM engine with the connector attached, generates a few
completions, then checks the workflow and task records that reach Flowcept.

Multiprocessing is disabled so the Flowcept buffer is readable in this process;
see the note in `run()` about why that also requires a drain request.

Run: python e2e_smoke.py [--model facebook/opt-125m]
"""

from __future__ import annotations

import argparse
import os

# Must precede the vLLM import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley.",
    "Write a haiku about recursion in programming:",
]

WORKFLOW_ID = "kvnorm-e2e-smoke"


def run(args) -> list[dict]:
    """Run the engine under a Flowcept context and return the captured buffer."""
    from flowcept import Flowcept
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    with Flowcept("vllm", workflow_id=WORKFLOW_ID, workflow_name="kvnorm_e2e_smoke") as fc:
        llm = LLM(
            model=args.model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="KVNormConnector",
                kv_connector_module_path="vllm_kvnorm",
                kv_role="kv_producer",
                kv_connector_extra_config={"workflow_id": WORKFLOW_ID},
            ),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
        outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24))
        print("\n=== generations ===")
        for o in outputs:
            print(f"  {o.request_id}: {o.outputs[0].text.strip()[:60]!r}")

        # Scoring happens one scheduler step after a request finishes. With
        # multiprocessing off nothing drives the loop once generate() returns,
        # so this trivial request supplies the step that flushes the last batch.
        llm.generate(["."], SamplingParams(temperature=0.0, max_tokens=1))
        del llm
        return list(fc.get_buffer())


def validate(buffer: list[dict], *, num_requests: int) -> int:
    print("\n=== validating Flowcept buffer ===")
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    workflows = [r for r in buffer if r.get("type") == "workflow" and r.get("conf")]
    tasks = [r for r in buffer if r.get("type") == "task"]

    check(len(workflows) == 1, f"exactly one model workflow (got {len(workflows)})")
    # The drain request produces a task too, so allow one extra.
    check(
        num_requests <= len(tasks) <= num_requests + 1,
        f"one task per request (got {len(tasks)} for {num_requests} prompts + drain)",
    )
    if not workflows or not tasks:
        return 1

    wf = workflows[0]
    conf = wf["conf"]
    for key in ("model", "tokenizer", "tokenizer_mode", "dtype", "max_model_len"):
        check(key in conf, f"workflow conf carries {key!r}")
    check(
        wf.get("parent_workflow_id") == WORKFLOW_ID,
        f"run nests under the caller's workflow (got {wf.get('parent_workflow_id')!r})",
    )
    check(wf["workflow_id"] != WORKFLOW_ID, "run has its own workflow id, not the caller's")

    task = tasks[0]
    check(task["workflow_id"] == wf["workflow_id"], "tasks reference the model workflow")
    check(task["status"] == "FINISHED", "task marked FINISHED")
    check(task["activity_id"] == "kv_token_importance", "activity_id set by the adapter")

    meta = task.get("custom_metadata", {})
    check(meta.get("metric") == "pagedeviction_v_over_k_l2", "metric label recorded")
    check(meta.get("num_layers", 0) > 0, f"layers captured ({meta.get('num_layers')})")

    gen, used = task["generated"], task["used"]
    check(list(gen) == ["score"], f"generated holds exactly one field, got {list(gen)}")
    scores = gen["score"]
    check(all(isinstance(s, (int, float)) and s == s for s in scores), "all scores finite")
    check(all(s > 0 for s in scores), "all scores positive (V/K ratio)")

    n = used["num_computed_tokens"]
    check(len(scores) == n, f"exactly one score per token ({len(scores)} vs {n})")
    check(meta.get("num_tokens") == n, "num_tokens matches computed tokens")

    prompt = used.get("prompt_token_ids")
    check(isinstance(prompt, list) and bool(prompt), "prompt_token_ids captured")
    check(
        len(prompt) == used["num_prompt_tokens"],
        f"prompt length matches num_prompt_tokens ({len(prompt)} vs {used['num_prompt_tokens']})",
    )
    check(all(isinstance(t, int) for t in prompt), "prompt ids are ints")

    print(f"\nFAILURES: {len(failures)}")
    for f_ in failures:
        print("  -", f_)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    args = ap.parse_args()
    return validate(run(args), num_requests=len(PROMPTS))


if __name__ == "__main__":
    raise SystemExit(main())
