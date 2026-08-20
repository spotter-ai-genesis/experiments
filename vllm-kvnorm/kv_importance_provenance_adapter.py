"""
Captures per-token KV-cache importance scores from vLLM as Flowcept provenance
via the "vllm" adapter (flowcept/src/flowcept/flowceptor/adapters/vllm/).

`run_workflow()` below has zero Flowcept API in its body -- no FlowceptTask,
FlowceptLoop, or @flowcept_task, only the `with Flowcept(...)` context plus one
`kv_transfer_config` entry telling vLLM to load the KVNormConnector. Every
completed request then transparently emits a Flowcept task carrying one
importance score per token.

The score is the PagedEviction proxy (arXiv:2509.04377): mean over layers and
KV heads of ||V_i||_2 / ||K_i||_2. It needs no attention weights, hence no
FlashAttention kernel changes.

Process model note: vLLM runs its scheduler in a separate EngineCore process, so
the interceptor lives there and publishes over Flowcept's MQ, exactly as the
Dask adapter does for workers. That means a live MQ (Redis) is required unless
you force everything into one process with VLLM_ENABLE_V1_MULTIPROCESSING=0,
which is what this example does so it runs with no services.

The vLLM run registers its own workflow, nested under this script's via
parent_workflow_id, because Flowcept records a given workflow once and reusing
the caller's id would drop the model/tokenizer configuration.

Run: python kv_importance_provenance_adapter.py
"""

import os
from pathlib import Path

# Must be set before vLLM is imported. See the process-model note above.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# Base name only: depending on `project.n.append_workflow_id_to_path` /
# `append_id_to_path`, dump_buffer() suffixes the workflow and writer ids onto
# this, so the file on disk may be e.g. *_<workflow_id>_<pid>_<tid>.jsonl.
# read_buffer_file() resolves that for us given the same base path.
BUFFER_PATH = OUTPUT_DIR / "flowcept_buffer_kv_importance.jsonl"

from flowcept import Flowcept  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

MODEL = "facebook/opt-125m"
WORKFLOW_ID = "kv-importance-demo"

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns.",
    "Write a haiku about recursion in programming:",
]


def build_llm() -> LLM:
    """A normal vLLM engine, plus the connector that does the capture."""
    return LLM(
        model=MODEL,
        kv_transfer_config=KVTransferConfig(
            kv_connector="KVNormConnector",
            kv_connector_module_path="vllm_kvnorm",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "sink": "flowcept",
                "workflow_id": WORKFLOW_ID,  # parent; the run nests under it
            },
        ),
        max_model_len=512,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
    )


def run_workflow(llm: LLM) -> None:
    """No Flowcept API here -- capture is transparent."""
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24))
    for out in outputs:
        print(f"  {out.request_id}: {out.outputs[0].text.strip()[:60]!r}")

    # Scores are computed one scheduler step after a request finishes. With
    # multiprocessing enabled the EngineCore loop keeps stepping on its own
    # while has_pending_push_work() is true, so the tail drains by itself. Here
    # multiprocessing is off (so the buffer is readable in-process), which means
    # nothing drives the loop once generate() returns -- this trivial request
    # supplies the step that flushes the last batch.
    llm.generate(["."], SamplingParams(temperature=0.0, max_tokens=1))


def report(buffer) -> None:
    """Show what the adapter captured, and decode a prompt to prove the join.

    Reads back the dumped buffer file rather than the in-memory one, so this
    also proves the provenance survives a round trip to disk.
    """
    workflows = [r for r in buffer if r.get("type") == "workflow" and r.get("conf")]
    tasks = [r for r in buffer if r.get("type") == "task"]

    print(f"\ncaptured {len(tasks)} tasks under {len(workflows)} model workflow(s)")
    if not workflows or not tasks:
        return

    conf = workflows[0]["conf"]
    print(f"  model     : {conf['model']}")
    print(f"  tokenizer : {conf['tokenizer']}")
    print(f"  parent    : {workflows[0].get('parent_workflow_id')}")

    task = tasks[0]
    scores = task["generated"]["score"]
    ids = task["used"]["prompt_token_ids"]
    print(f"\n  {task['task_id']}: {len(scores)} scores, {len(ids)} prompt tokens")

    # The tokenizer recorded on the workflow is what makes the ids decodable.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(conf["tokenizer"])
        print("  prompt token importance:")
        for i, piece in enumerate(tok.convert_ids_to_tokens(ids)):
            print(f"    {scores[i]:.4f}  {piece!r}")
    except Exception as exc:  # tokenizer is optional for the demo
        print(f"  (skipped decoding: {exc})")


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for stale in OUTPUT_DIR.glob(f"{BUFFER_PATH.stem}*.jsonl"):
        stale.unlink()

    with Flowcept("vllm", workflow_id=WORKFLOW_ID, workflow_name="kv_importance_demo"):
        llm = build_llm()
        run_workflow(llm)
        del llm  # flushes the connector
        Flowcept.get_current_instance().dump_buffer(str(BUFFER_PATH))

    # consolidate=True merges the per-writer files dump_buffer() produced (it
    # suffixes pid/thread when append_id_to_path is set) and, with
    # cleanup_files, removes them -- leaving one file holding the run.
    records = Flowcept.read_buffer_file(
        str(BUFFER_PATH), consolidate=True, workflow_id=WORKFLOW_ID, cleanup_files=True
    )

    written = sorted(OUTPUT_DIR.glob(f"{BUFFER_PATH.stem}*.jsonl"))
    for f in written:
        print(f"\nbuffer written to {f} ({f.stat().st_size} bytes)")

    report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
