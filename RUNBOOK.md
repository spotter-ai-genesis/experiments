# Runbook — running the OPAL poisoning suite

Start to finish. Steps 1–2 are on your laptop, 3–7 on the GPU box, 8 back on
the laptop. If you only remember one thing: **step 5 exists because a dead
capture is not self-announcing.**

For what the experiment *is*, see [OPAL.md](OPAL.md). This file is only
how to run it.

---

## 1. Point at the box — once per shell

```bash
cd /home/agueroudji/Work/genesis/experiments_10_26_26

export OPAL_BOX=agueroudji@<gpuhost>
export OPAL_BOX_PATH=genesis/experiments/experiments_10_26_26
export OPAL_SSH_OPTS='-o ControlMaster=auto -o ControlPath=/tmp/opal-%r@%h:%p -o ControlPersist=4h'
```

Put these in `~/.bashrc`. `OPAL_BOX_PATH` failing to survive a new terminal is
the single most common way `pull` fails, and rsync's own error names a path
nobody chose. `OPAL_SSH_OPTS` reuses one connection so you are not
re-authenticating on every call.

## 2. Push — no git involved

```bash
./scripts/sync_box.sh push -n     # dry run: look at the list first
./scripts/sync_box.sh push
```

About 1.6 MB. Push is an explicit include-list, so a venv, a vendored clone or
a run output never ships by accident. `bench/opal/data/poisoned/` is
deliberately **not** sent — the box regenerates it, because token parity is
tokenizer-dependent and a stale CSV is wrong silently.

## 3. Set up the box — once, ever

```bash
ssh $OPAL_BOX
cd genesis/experiments/experiments_10_26_26
./scripts/setup_box.sh
```

It installs `uv` if absent and builds `.venv` on a uv-managed CPython 3.12,
because the box's own python3 has neither `ensurepip` nor `Python.h` and both
need root to add. A `.venv` left over from a failed attempt is discarded
rather than reused. `OPAL_SYSTEM_PYTHON=1` opts out,
`OPAL_PYTHON_VERSION=3.11` picks another version.

**Read the last block it prints.** Two lines decide whether anything that
follows is real:

```
connector  .../vllm-attn-connector (hybrid-multimodal-kv-support)
probe installs: True
```

Setup now **refuses to continue** if the connector lacks hybrid-KV support,
checked by reading its source rather than trusting the branch name — a
checkout already on the box is reused on whatever branch it is on, and
`--branch` only applies when this script clones. Qwen3.5 is a hybrid model
(gated-delta-net plus full attention, 10 of 40 layers) and off that support
the connector finds no block table for the scored layers and captures
**zeros** with no error: valid records, correct shapes, every value 0.0.

If it stops there, the message gives you the three ways out — checkout the
branch, let setup use its own clone in `vendor/`, or apply
`vendor-patches/vllm-attn-connector-hybrid-kv.patch`.

Your `flowcept` and `vllm-attn-connector` checkouts already live next door on
that box, so setup reuses them and **will not edit them**. If either needs the
PEP 639 or NVML fix it prints the one-line `sed` and leaves the call to you.

## 4. Pre-flight — no GPU touched

```bash
DRY=1 ./scripts/run_suite.sh
```

Builds all 9 poisoned CSVs, pins every arm to the common core, and plans all
ten arms. You want:

- the pinned list, **10 chains**, printed once
- `ok` on all ten arms

This re-checks token parity against the box's tokenizer, so a manifest that
does not balance aborts here in seconds rather than an hour into capture.

## 5. Smoke test — do not skip this

```bash
OPAL_LIMIT=5 ./scripts/run_suite.sh clean qc
```

At `OPAL_LIMIT=5` the common core is 1 chain — enough to prove the capture
works, which is all this step is for. Below 5 the arms share nothing and the
pin has nothing to pin to; use `OPAL_PIN=0` there.

Read the `=== capture verification ===` block. Every line must be PASS, and
one of them matters more than the rest:

```
PASS  the capture is NON-ZERO
```

That is the whole reason this step exists. A dead capture still writes
well-formed records whose every number is zero. Five runs were reduced,
plotted and believed before anyone noticed.

## 5b. Shared box? Cap the reservation

`--gpu-memory-utilization` is a fraction of **total** memory, not free memory.
On a unified-memory part the GPU and host share one pool, so the default 0.88
of a 122 GiB box reserves ~107 GiB — out of the same pool another user is in.
The pre-flight refuses to start in that case and suggests a value. Pass it:

```bash
OPAL_EXTRA_ARGS='--gpu-memory-utilization 0.36 --max-model-len 24576' \
  ./scripts/run_suite.sh
```

Your model's weights must still fit inside the capped reservation. Rough bf16
footprints: Qwen3.5-4B ~8 GiB, 9B ~18 GiB, 27B ~54 GiB, 35B-A3B ~67 GiB; an
FP8 quantisation is roughly half.

## 6. The full run

Check free disk first.

```bash
df -h .
nohup ./scripts/run_suite.sh > suite.log 2>&1 &
tail -f suite.log
```

`nohup` because a dropped ssh session takes its children with it and this is
long. The first run for a given model pays a one-off FlashInfer JIT build;
`MAX_JOBS=2` keeps it from taking the box down, and the result is cached so
later runs skip it. Clean plus 9 arms, sequential on one card, identical engine settings,
T=0, `--max-tokens 16384`, 10 pinned chains = 57 turns per arm.

**Disk:** measured 0.83 GiB (clean) and 1.94 GiB (poisoned) per arm at 90
turns; pinned runs are 57 turns, so budget roughly **5–12 GB** for the suite.
Poisoned arms are consistently larger because the model reasons longer.

If an arm dies the script says so and carries on — the others stay valid.

## 7. Reduce on the box

```bash
python scripts/attn_batch.py
```

Writes `analytics-out/<run>/` — combined CSV, per-chain CSVs, and a text heat
map per chain. A run with an empty capture is listed as **skipped**, not
silently dropped.

Do this **here**, not at home: `analytics-out/` is a few MB against 5–12 GB of
tensors.

## 8. Pull and read

```bash
exit                              # back on the laptop
./scripts/sync_box.sh ls          # what runs exist on the box
./scripts/sync_box.sh pull        # EVERYTHING, tensors included
```

That brings back whole run directories (`turns.jsonl`, `summary.json`,
`run_meta.json`, `provenance.jsonl`, `suite_arm.txt`, `poison_manifest.json`
and `attention/`), plus `analytics-out/` and the poison manifests. A suite is
10–20 GB — check disk first. To read results without the tensors:

```bash
./scripts/sync_box.sh pull --light
./scripts/sync_box.sh pull 20261027T223905Z-qc     # one run, everything
```

Then:

```bash
for f in runs/*/summary.json; do echo "== $f"; python -m json.tool "$f" | head -20; done
less analytics-out/<run>/renders/exp32-chain-0020-hard.txt
```

---

## Read the numbers with these in hand

- **Check `truncated` per arm before anything else.** Truncation is
  arm-dependent: the poisons that work also make the model ramble past the
  cap. At 8192 the `qc` arm lost 33 of 90 turns against 2 for clean, which is
  a confound pointing the same way as the result. The suite uses 16384, where
  it loses 11 and accuracy is unchanged — that is how you know the effect is
  real, and it is why the cap is not a knob to turn down for speed.
- **All arms are pinned to the same 10 chains**, every one of which is
  poisoned in every arm, so cross-arm numbers compare directly. With 10 chains
  the per-arm statistics are thin — a weak arm will not separate from clean.
  `python scripts/common_chains.py --report` shows the coverage behind the pin.
  The pool size is the lever, measured on this benchmark:

  | `OPAL_LIMIT` | 5 | 10 | 15 | 20 | 30 |
  |---|---|---|---|---|---|
  | chains in the core | 1 | 4 | 7 | **10** | 14 |

  The pin caps what actually runs, so a wider pool costs nothing per arm
  beyond the extra chains you gain.
- **Ground truth moves under some arms, on purpose.** An arm that legitimises
  a documented ablation carries a truthful `correct_answer`, so a model that
  reasons correctly over poisoned tables scores RIGHT. Rounds after the first
  are not re-derived and are marked `ground_truth_invalidated`.
- **`history` is not in the suite** — it rewrites the assistant's reply, which
  a CSV cannot express. To run it:

  ```bash
  python scripts/build_runtime_manifest.py --model Qwen/Qwen3.5-35B-A3B --limit 20
  python scripts/run_experiment.py --poison configs/poison/history-runtime.json \
    --csv bench/opal/data/opal_chains.csv --model Qwen/Qwen3.5-35B-A3B \
    --enable-thinking --max-tokens 16384 --max-model-len 32768 --temperature 0.0 \
    --max-num-seqs 8 --cuda-graphs --out-dir runs/history
  ```

## When it goes wrong

| symptom | cause |
|---|---|
| `capture is NON-ZERO` FAILS | `VLLM_ENABLE_V1_MULTIPROCESSING` unset — `run_suite.sh` exports it, a bare `run_experiment.py` does not. (The connector's hybrid-KV support is now gated at setup, so it should not reach here.) |
| setup stops at `does not support hybrid KV caches` | the connector checkout predates hybrid support; the message lists the three fixes |
| **the box falls over / OOM killer / ninja "exit status 137"** | `MAX_JOBS` unset. FlashInfer JIT-builds the MoE kernel at warmup and ninja spawns one nvcc per core (~20); on a unified-memory part they compete with the resident weights and the machine goes down. `run_suite.sh` exports `MAX_JOBS=2` (measured safe with a 57 GiB MoE model). Running `run_experiment.py` by hand? Export it yourself |
| `FileNotFoundError: [Errno 2] ... 'ninja'` deep in a flashinfer JIT frame | ninja is installed in `.venv/bin` but that directory was not on `PATH`. `run_suite.sh` now exports it and pre-checks the tool before any GPU work. If you invoke `run_experiment.py` by hand, export it yourself: `export PATH="$PWD/.venv/bin:$PATH"` |
| `No module named pip` / `Python.h is missing` | the system python3 is incomplete (Debian splits off `python3-venv` and `python3-dev`). `setup_box.sh` now builds on a uv-managed CPython, which has both; it also discards a `.venv` lacking either. `OPAL_SYSTEM_PYTHON=1` forces the system interpreter |
| push says `pushed` but the box dir is empty | re-push: the chmod step now verifies the payload landed and fails loudly instead of warning |
| `_capture_gpu ... NameError: name 'self' is not defined` | a bad early version of the NVML patch. `setup_box.sh` now detects and repairs it. Non-fatal: flowcept catches it, so the attention capture is unaffected — only GPU telemetry is lost |
| NVML `NotSupported` once per task | flowcept's unconditional memory probe on a unified-memory GPU; `setup_box.sh` patches clones it owns, and prints the fix for checkouts you own |
| `change_dir ... failed` on pull | `OPAL_BOX_PATH` unset in this shell |
| need to stop a run; Ctrl-C does nothing | `./scripts/stop_run.sh` to see it, `--kill` to stop it. Ctrl-C only reaches the foreground job, and a process in CUDA or `D` state ignores it. Do **not** kill a PGID derived from `ps \| grep run_experiment.py` — that also matches the shell that launched it, and killing that group logs you out |
| hangs after `torch.compile took N s` | CUDA-graph capture, which re-touches every weight. If the checkpoint is bigger than RAM (vLLM says so at load) it page-faults off disk and can take very long. `OPAL_EAGER=1` skips capture — right for a smoke test, costs decode throughput on the full suite |
| `FATAL: vLLM would reserve N GiB but only M GiB is free` | the pre-flight memory guard. `--gpu-memory-utilization` is a fraction of TOTAL memory, not free memory, and on a unified-memory box that pool is shared with other users — the default 0.88 of 122 GiB reserves 107 GiB and can take someone else's job down. Use the value the message suggests |
| `No available memory for the cache blocks` | weights (~67 GiB) leave too little at `--gpu-memory-utilization 0.88`. The cheap fix is the window, not the utilization: longest pinned prompt is 5067 tokens, so `OPAL_EXTRA_ARGS='--max-model-len 24576'` is safe and returns ~25% of the cache. Do **not** lower `--max-tokens`; that is what makes truncation arm-dependent |
| pre-flight aborts on parity | a strategy does not balance under this tokenizer; `scripts/make_poison_csv.py --probe` shows the deltas |
