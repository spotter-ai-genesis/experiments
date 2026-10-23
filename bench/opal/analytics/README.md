# Per-segment prefill attention

Two scripts, split so that capture needs a GPU and inspection does not:

| script | does | needs |
| --- | --- | --- |
| `attn_analytics.py` | runs a chain, captures attention, writes a CSV | GPU, vLLM, flowcept |
| `render_heatmap.py` | draws a CSV as a text heat map | just the tokenizer |

`attn_analytics.py` runs one OPAL chain through vLLM, captures exact
per-decode-step attention over the prompt with
[`vllm-attn-connector`](../../../../vllm-attn-connector), and reduces it to
**one attention vector per reply segment**.

A round's answer is a list of items separated by `;` — `NIL 373; DIVERGES` is
two. Each item is produced by its own run of decode steps. For prefill token
`j` the score for an item is

```
score[j] = Σ_i  0.5^i · val_all_max[step_i][j]
```

summed over that item's content steps, with `i` counted from the item's first
step. The decay weights the step that *commits* to the item above the steps
that merely spell it out. Tokens outside the connector's top-`--top-pct`
selection score 0.

The `ANSWER:` prefix and the `;` separators belong to no item and are excluded.

---

## 1. Prerequisites

**Hardware.** A local CUDA GPU with **8 GB VRAM**. Adjustments needed: the
memory flags in §4 are sized for 8 GB. With more VRAM you can drop
`--max-num-batched-tokens` and raise `--max-model-len`; with less, lower
`--max-model-len` and `--top-pct`, or use a smaller model.

**Repos.** `vllm-attn-connector` and `flowcept` must sit beside `experiments/`
under `SPOTTER-AI/`. The script locates the connector by walking up from its own
path, so it needs no configuration.

## 2. One-time setup

### 2a. Create the Python environment

Build a [uv](https://docs.astral.sh/uv/) virtualenv and install vLLM into it.
vLLM pulls in torch, transformers, safetensors and numpy, so it goes first and
determines the CUDA build you get.

```bash
export SPOTTER=/path/to/SPOTTER-AI   # repo root; used throughout this README

cd "$SPOTTER"                        # anywhere you like, the venv is not repo-bound
uv venv spotter-ai
source spotter-ai/bin/activate
uv pip install vllm
```

Then add the packages flowcept needs, which a stock vLLM environment does not
have. All are small and pure Python — none disturb the torch/CUDA stack:

```bash
uv pip install omegaconf msgpack orjson rich
```

Check what you ended up with:

```bash
python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Validated against Python 3.14.8, torch 2.13.0+cu130, vLLM 0.32.0. Other
combinations should work, but the connector attaches to vLLM's attention
backend and KV-connector APIs, so a major vLLM bump may need it updated.

Every `python` below assumes this venv is active. If you prefer not to
activate, call its interpreter directly (`spotter-ai/bin/python`) and pass
`--python spotter-ai/bin/python` to `uv pip`.

### 2b. Create a flowcept settings file

Flowcept reads its configuration from `FLOWCEPT_SETTINGS_PATH`. Copy the
template into the gitignored scratch area — **do not** edit
`~/.flowcept/settings.yaml`:

```bash
cd "$SPOTTER/flowcept"
mkdir -p agent_sandbox
cp -n resources/sample_settings.yaml agent_sandbox/settings.yaml
```

The defaults are already what this needs: `mq.enabled: false` and
`db_flush_mode: offline`, so **no Redis and no MongoDB are required**.
Provenance is buffered in memory and the attention tensors are written straight
to disk as SafeTensors.

### 2c. Fetch the model weights

Both are ungated 4-bit builds — the official `meta-llama` repo is gated and
returns 401 without a token, so the Llama weights come from `hugging-quants`.
FP16 Llama 3.1 8B needs ~16 GB and will **not** fit an 8 GB card.

```bash
hf download cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit
hf download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

About 3.2 GB and 5.3 GB respectively. First run takes a few minutes; afterwards
the engine loads in under a second.

## 3. Environment for each run

Only `flowcept` needs to be on `PYTHONPATH` — the script inserts the connector's
`src/` and `tests/` itself.

```bash
cd "$SPOTTER/experiments/bench/opal/analytics"
export PYTHONPATH="$SPOTTER/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="$SPOTTER/flowcept/agent_sandbox/settings.yaml"
```

`$SPOTTER` was set in §2a. In a fresh shell, set it again before the above:
`export SPOTTER=/path/to/SPOTTER-AI`.

`VLLM_ENABLE_V1_MULTIPROCESSING=0` is required but the script sets it itself.

## 4. Run `exp32-chain-0020-hard`

Both commands use `--ground-truth-history`, which teacher-forces each round with
the correct prior answer. That is what you want when studying **a single round's
evidence**: without it an early mistake poisons later rounds and you end up
measuring error propagation instead. Drop the flag to study propagation.

### Qwen3-4B (AWQ 4-bit)

```bash
python attn_analytics.py \
    ../data/opal_chains.csv \
    --chain-id exp32-chain-0020-hard \
    --ground-truth-history \
    --out output/attn_exp32-chain-0020-hard_qwen3-4b_gt-history.csv \
    --prov-dir /tmp/aseg_qwen
```

### Llama-3.1-8B (AWQ INT4)

```bash
python attn_analytics.py \
    ../data/opal_chains.csv \
    --chain-id exp32-chain-0020-hard \
    --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --ground-truth-history \
    --out output/attn_exp32-chain-0020-hard_llama-3.1-8b_gt-history.csv \
    --prov-dir /tmp/aseg_llama \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.93
```

**`--max-num-batched-tokens` is not optional on an 8 GB card.** vLLM profiles
with a dummy batch of 8192 tokens by default and materialises a `batch × vocab`
logits tensor. At Llama's 128k vocab that is ~2.1 GiB on top of 5.3 GiB of
weights, and the engine OOMs before generating anything. Qwen survives the
default only because its weights are 2 GB smaller.

Each run takes roughly 60–100 seconds once weights are cached — generation
itself is under 3 seconds, the rest is engine startup. Run in the background if
your shell may time out.

## 5. Expected output

```
  round 1 dose_response_shape     got='100'  [953 tok, 8 steps]
  round 2 transition_sensitivity  got='NIL 373'  [1549 tok, 9 steps]
  round 3 cross_modality_peak     got='100; AGREES'  [2307 tok, 11 steps]
  round 4 modality_coverage       got='rgb2,msc1,vnir,swir,s3d'  [2951 tok, 19 steps]
  round 5 analogue_search         got='NIL 373; DIVERGES'  [3534 tok, 13 steps]
  round 6: skipped, 4471 tokens exceeds the window
wrote 7 rows -> output/attn_..._qwen3-4b_gt-history.csv
```

Qwen stops at round 5: its round-6 prompt is 4471 tokens against the 4096
window. Llama tokenizes the same chain more compactly (4024) and completes all
six, writing 8 rows. Raising `--max-model-len` for Qwen needs more VRAM than
this card has.

### The CSV

One row per `(round, segment)`, in column order:

| column | meaning |
| --- | --- |
| `chain_id`, `round_no`, `task_type` | which round |
| `model` | the model that produced it |
| `segment_index` | 0-based position of the item within the answer |
| `segment_text` | the item, e.g. `NIL 373` or `DIVERGES` |
| `decode_steps` | the decode steps that produced it |
| `n_prompt_tokens` | prompt length for that round |
| `prompt_token_ids` | that prompt, so the CSV renders without the chain file |
| `scores` | space-separated float **per prefill token**, same length and order as the prompt |

The two CSVs in `output/` are produced by exactly the §4 commands, and are what
the `render_heatmap.py` examples below read.

### Rendering a heat map

`render_heatmap.py` is separate and reads **only the CSV** — no GPU, no vLLM,
no chain file, just the tokenizer named in the `model` column:

```bash
python render_heatmap.py output/attn_exp32-chain-0020-hard_qwen3-4b_gt-history.csv --round 5
python render_heatmap.py output/attn_..._qwen3-4b_gt-history.csv --all --width 110 > round5.txt
```

One column per reply segment. Each line's cell is the max over that line's
tokens, scaled to its own column's maximum, on the ramp `' .:-=+*#%@'`. Columns
are scaled independently, so they show *where* an item looked, not how much
attention it drew relative to another item.
