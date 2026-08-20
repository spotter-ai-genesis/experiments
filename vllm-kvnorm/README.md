# vllm-kvnorm tests

Tests for the `vllm_kvnorm` connector, which captures one importance score per
token from vLLM's paged KV cache when a request finishes, using the PagedEviction
proxy `mean(||V||_2 / ||K||_2)` ([arXiv:2509.04377](https://arxiv.org/abs/2509.04377)).

## Setup

Assumes vLLM is already installed. Two more things must be importable: the
connector (`../../vllm-kvnorm`) and Flowcept with its `vllm` adapter
(`../../flowcept`) — a released Flowcept will not carry that adapter, so it has
to come from the local checkout.

```bash
ROOT=$(cd ../.. && pwd)                 # .../20B-SPOTTER-AI
```

**Not editing either package** — install them:

```bash
pip install "$ROOT/flowcept" "$ROOT/vllm-kvnorm"
```

**Editing either package** — put them on the path instead, so edits take effect
without reinstalling:

```bash
export PYTHONPATH="$ROOT/vllm-kvnorm/src:$ROOT/flowcept/src"
```

A plain `pip install` snapshots the source, so while editing you would silently
keep testing the previously installed copy. `PYTHONPATH` is inherited by vLLM's
EngineCore subprocess, so the connector resolves there too. (`pip install -e`
also works if you prefer editable installs.)

Either way, point Flowcept at a settings file:

```bash
export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
```

All scripts set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the Flowcept buffer is
readable in-process and no MQ is needed.

The tests assert against the in-memory buffer and write nothing. The example
additionally dumps its buffer to `output/` (gitignored) via
`Flowcept.dump_buffer()`, following the `vit/` example. Note that with
`append_workflow_id_to_path` / `append_id_to_path` enabled in settings, the file
on disk is suffixed per writer — read it back with
`Flowcept.read_buffer_file(base, consolidate=True, workflow_id=...)`.

## Running

```bash
pytest test_kernels.py -q              #  3 s, no GPU engine
pytest test_incremental_scoring.py -q  #  7 s, GPU but no engine
python e2e_smoke.py                    # ~1 min
python test_preemption.py              # ~1 min
```

The engine-backed scripts take `--model` (default `facebook/opt-125m`, ~250 MB,
downloaded on first run). All print `FAILURES: 0` and exit non-zero on failure.

| Script | What it checks |
|---|---|
| `test_kernels.py` | Triton kernel vs torch reference across head sizes/counts; all four KV layouts; one score per token; TP reduction maths |
| `test_incremental_scoring.py` | **Step-by-step scoring equals one-shot** — under chunked prefill, incrementally arriving blocks, preemption restarts, concurrent requests |
| `e2e_smoke.py` | Real vLLM run; the Flowcept workflow + task records match the documented schema |
| `test_preemption.py` | Under forced preemption the hook fires once per *request*, covering the full final sequence |

`test_incremental_scoring.py` is the load-bearing one, because incremental
accumulation is where the bugs are. An early version relied on the kernel's
`out=` argument to accumulate across layers, but it *stores* — only the last
layer survived and every score was ~100% wrong, while every end-to-end check
still passed because they only compared the connector against itself.
Reintroducing that bug fails 6 of the 8 cases.

Plus a runnable Flowcept example:

```bash
python kv_importance_provenance_adapter.py
```

Shows the capture end to end and decodes the prompt using the tokenizer
recorded on the workflow. Adapter unit tests live in
`flowcept/tests/adapters/test_vllm.py`.

## Other models

```bash
python e2e_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Validated on `facebook/opt-125m` (MHA, 12 layers, 12 KV heads) and
`Qwen/Qwen2.5-0.5B-Instruct` (GQA, 24 layers, 2 KV heads). Needs ~6 GB VRAM.

`e2e_smoke.py` takes `--model`, `--max-model-len` and `--gpu-memory-utilization`.
There are no scoring knobs: the output is always one float per token.
