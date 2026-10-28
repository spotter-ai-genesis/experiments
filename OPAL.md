# OPAL attention-provenance experiment

**To run it, follow [RUNBOOK.md](RUNBOOK.md).** This file is what the
experiment is and why it is built this way.

One clean run and one run per poisoning strategy, on Qwen3.5, with Flowcept
provenance and per-token attention capture — then the reduction in
`analytics/` turns each run into one scored attention vector per reply item.

Everything needed is in this directory.

## Two machines

| | |
|---|---|
| **Laptop (CPU)** | build the datasets, pre-flight a run, reduce and render a finished capture |
| **GPU box** | the runs themselves — needs vLLM, the attention connector and Qwen3.5 |

**GPU box** — one command, idempotent:

```bash
./scripts/setup_box.sh
```

It installs `uv` if absent, creates `.venv` on a uv-managed CPython (the
system python3 on this box has neither pip nor headers), installs vLLM, then
flowcept and
`vllm-attn-connector` **editable from source** against that vLLM, applies two
upstream fixes, writes `requirements.gpu.lock`, and verifies the probe
installs. Siblings already on the box are reused and never edited; missing
ones are cloned into `vendor/`.

The connector must support **hybrid KV caches**. Qwen3.5 is a hybrid model —
gated-delta-net plus full attention, 10 of 40 layers — and without that
support the connector finds no block table for the scored layers and captures
**zeros**, silently. `setup_box.sh` checks the connector source for it and
refuses to continue otherwise, rather than trusting a branch name: a checkout
already on the box is reused on whatever branch it happens to be on.

**Laptop** — datasets, pre-flight and the reduction only:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
```

## Getting it to the box

No git required:

```bash
export OPAL_BOX=agueroudji@gpuhost
export OPAL_BOX_PATH=genesis/experiments/experiments_10_26_26   # the default

./scripts/sync_box.sh push            # ~1.5 MB: code + the clean CSV
./scripts/sync_box.sh push -n         # dry run first, if you like
```

Push is an explicit include-list, not an exclude-list, so a new venv or
vendored clone never ships by accident. `bench/opal/data/poisoned/` is deliberately not
sent — `scripts/run_suite.sh` regenerates it on the box, because token parity is
tokenizer-dependent and a stale CSV is wrong silently.

Coming back:

```bash
./scripts/sync_box.sh ls              # what runs exist there
./scripts/sync_box.sh pull            # results WITHOUT the tensors
./scripts/sync_box.sh pull --attention            # with them
./scripts/sync_box.sh pull --attention <run-name> # one run, with them
```

`pull` leaves the tensors behind by default: one arm's `attention/` is
0.8–2.0 GiB and a full suite is 10–20 GB. What you get without them —
`turns.jsonl`, `summary.json`, `run_meta.json`, `provenance.jsonl` and
`analytics-out/` — is enough to read every result and every heat map. You need
`--attention` only to run `scripts/attn_batch.py` locally, and reducing on
the box then pulling `analytics-out/` is far cheaper.

## Run everything

```bash
DRY=1 ./scripts/run_suite.sh        # pre-flight all arms, no GPU touched
./scripts/run_suite.sh              # clean + every arm, in sequence
./scripts/run_suite.sh clean qc     # just these
OPAL_LIMIT=5 ./scripts/run_suite.sh # 5 chains instead of 20
```

Each run lands in `runs/<UTC timestamp>-<arm>/` with `turns.jsonl`,
`run_meta.json`, `summary.json`, `provenance.jsonl` and `attention/`.

Then:

```bash
python scripts/attn_batch.py
```

which writes `analytics-out/<run>/<run>.csv`, `chains/<chain>.csv` and
`renders/<chain>.txt`.

## An arm is a dataset, not a flag

`scripts/make_poison_csv.py` writes `bench/opal/data/poisoned/<arm>.csv` with the same
seven columns, the same rows in the same order and the same chain ids as
`bench/opal/data/opal_chains.csv`. Every run is therefore the same command differing only
in `--csv`, and nothing in `scripts/run_experiment.py` knows which arm it is executing.
That is what makes the clean arm a control rather than a different code path,
and it means a poisoned dataset can be diffed, versioned or handed to another
harness.

`scripts/run_suite.sh` regenerates the datasets on every invocation, because token
parity is model-dependent: a substitution that balanced under one tokenizer
can shift every index under another, and a stale CSV would do it silently.

### The arms

| arm | form | what it corrupts |
|---|---|---|
| `clean` | — | nothing (control) |
| `outfmt` | prose | the OUTPUT FORMAT instruction |
| `rules` | prose | a stated rule |
| `qc` | data-table | the quality-gate floor |
| `schema` | data-table | the schema dictionary |
| `null` | control | a session string that changes no answer |
| `alias_swap` | data-table | swaps plant-level and alias-row numbers |
| `floor_raise` | data-table | raises the usable-plant floor |
| `injection` | prose | an injected instruction |
| `relocate` | data-table | moves evidence within the prompt |

**`history` is not here.** It substitutes the *assistant's own reply*, and a
CSV holds no assistant turns, so it has no dataset form. It remains a runtime
manifest:

```bash
python scripts/run_experiment.py --poison configs/poison/history-runtime.json --csv bench/opal/data/opal_chains.csv ...
```

That manifest is tokenizer-specific, so rebuild it if you change model:

```bash
python scripts/build_runtime_manifest.py --model <model> --limit 20
```

### The correct answer moves, and that is the point

Lowering a quality gate does not merely mislead the model — it changes what
the right answer **is**. The benchmark records this already: `explanation`
names the answer each skipped step produces, so an arm that legitimises a
documented shortcut carries a truthful `correct_answer`. A model that reasons
correctly over poisoned tables now scores RIGHT and one that takes the
shortcut scores WRONG. The grading inverts, which is what makes the dataset a
test of the method rather than of the model.

Two limits, both stated in each `<arm>.manifest.json`:

- **Rounds after the first are not re-derived.** Round 2 says "starting from
  the dose you identified in round 1"; once round 1's answer changes, the
  stored answer for every later round describes a different investigation.
  Those rows pass through unchanged, marked `ground_truth_invalidated`.
- **Coverage is partial, and the arms do not cover the SAME chains.** An arm
  applies only where its anchor exists *and* the substitution is
  token-balanced: 20 chains for `outfmt` and `null`, 13 for the rest — and not
  the same 13. Only **10 are poisoned by every arm**.

  The suite therefore pins all arms, clean included, to that intersection, so
  every chain in the run is poisoned in every arm:

  ```bash
  python scripts/common_chains.py --report    # the coverage table
  python scripts/common_chains.py --oneline   # just the pinned ids
  OPAL_PIN=0 ./scripts/run_suite.sh           # opt out, run all 20
  ```

  Without the pin, a chain an arm never touched scores as "the poison did
  nothing" — indistinguishable from "applied, and resisted". That biases every
  arm toward looking weaker, and unevenly, because the arms miss different
  chains. Opt out only if you intend to do the subsetting in the analysis.

## Adding a strategy

Each arm is documented in [METHODS.md](METHODS.md) — what it edits, why, and
what it measured.


Drop a JSON file in `configs/strategies/`; `--all` picks it up. Author it with
the probe, which reports the token delta of every candidate phrasing — parity
rejects most wordings and will not tell you which unless you ask:

```bash
python scripts/make_poison_csv.py --probe --strategy configs/strategies/mine.json
```

A strategy needs `find`/`replace` (or a candidate list), the round it applies
to, and `legitimises` naming the ablation from `explanation` whose answer
becomes the new truth — omit it and the arm grades obedience, not correctness.

## Traps

- **Character-length-preserving is not token-length-preserving.**
  `AGREES`→`DENIES` is 6 characters both ways and shifts every later index.
  Parity is enforced per chain and an unbalanced substitution is refused.
- **A dead capture is not self-announcing.** `scripts/run_experiment.py` verifies that
  the probe fired, covered every scored layer, is non-zero, and that attention
  sums to one softmax unit per decode step. Five runs were once reduced and
  believed before anyone noticed there was no attention in them.
- **Truncation is arm-dependent, so it is a confound.** The poisons that work
  also make the model ramble past the cap. At `--max-tokens 8192` the `qc` arm
  lost 33 of 90 turns against 2 for clean. The suite uses 16384, where it
  loses 11 and accuracy is unchanged — which is how you know the effect is
  real.
- **Never score the raw completion.** A thinking block contains the model's own
  prose "Answer:", and the answer regex is case-insensitive and
  last-match-wins. Score the visible reply; `tests/test_scoring.py` pins this.
- **Never reuse a run directory.** Flowcept keeps old dumps and names them with
  a pid. `scripts/run_suite.sh` timestamps every run.

## Layout

```
scripts/
  run_suite.sh            the launcher: clean + every arm
  stop_run.sh             stop a running suite, release the GPU
  run_experiment.py       one run against one CSV
  setup_box.sh            one-time GPU box setup
  sync_box.sh             push code / pull results over ssh, no git
  make_poison_csv.py      builds bench/opal/data/poisoned/<arm>.csv + manifest
  common_chains.py        the chains every arm poisons; the suite pins to these
  build_runtime_manifest.py   rebuilds the history manifest for a new model
  attn_from_run.py        reduces ONE captured run
  attn_batch.py           reduces every run under runs/
configs/
  flowcept_gpu.yaml       offline provenance, GPU telemetry on
  strategies/*.json       declarative poisoning strategies
  poison/history-runtime.json   the one arm with no dataset form
bench/opal/               THE BENCHMARK, and nothing else
  data/opal_chains.csv    the chains
  data/poisoned/          generated, one CSV + manifest per arm
  analytics/              upstream reduction + renderer, kept byte-identical
src/opal_attn/            chains, prompts, ranges, runner, scoring, capture checks
runs/                     one directory per run
analytics-out/            the reduction, one directory per run
tests/                    CPU-only
vendor-patches/           connector patch for Qwen3.5 hybrid KV
```
