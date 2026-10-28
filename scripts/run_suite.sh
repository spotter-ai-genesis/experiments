#!/bin/bash
# THE launcher: the clean arm, then one run per poisoning strategy.
#
#   ./scripts/run_suite.sh              # clean + every arm
#   ./scripts/run_suite.sh clean qc     # just these
#   DRY=1 ./scripts/run_suite.sh        # pre-flight only, no GPU touched
#   OPAL_LIMIT=5 ./scripts/run_suite.sh # consider 5 chains instead of 20
#   OPAL_PIN=0 ./scripts/run_suite.sh   # do NOT pin to the common core
#   OPAL_EAGER=1 ./scripts/run_suite.sh # skip CUDA-graph capture (fast start)
#   OPAL_EXTRA_ARGS='--gpu-memory-utilization 0.94 --max-model-len 24576' \
#     ./scripts/run_suite.sh            # anything else, without editing this
#
# An arm is a POISONED COPY OF THE CSV, not a flag. make_poison_csv.py writes
# bench/opal/data/poisoned/<arm>.csv with the same seven columns and the same chain ids,
# so every run here is the same command differing only in --csv. Nothing in
# run_experiment.py knows which arm it is executing, which is what makes the
# clean arm a real control rather than a different code path.
#
# EVERY ARM RUNS THE SAME CHAINS, AND EVERY ONE OF THEM IS POISONED IN EVERY
# ARM. An arm poisons a chain only where its anchor exists AND the
# substitution is token-balanced, so the arms cover different subsets -- 20
# chains for outfmt and null, 13 for the rest, and not the same 13. Compared
# across all 20, a chain an arm never touched scores as "the poison did
# nothing", which is indistinguishable from "applied, and resisted"; that
# biases every arm toward looking weaker, and unevenly. So the suite pins all
# arms, clean included, to the intersection (scripts/common_chains.py).
# OPAL_PIN=0 turns this off and runs the full set, which is the right choice
# only if you intend to do the subsetting in the analysis instead.
#
# WHY SEQUENTIAL. vLLM's numerics depend on batch composition, so two arms
# scheduled differently can disagree on a few answers for reasons that have
# nothing to do with the poison. Every arm gets identical --max-num-seqs,
# --max-model-len, --limit and model, back to back with nothing else on the
# card. Holding that by hand across ten invocations is how it stops being held.
#
# T=0.0 ON PURPOSE. At T>0 a clean/poisoned pair differs by sampling as well as
# by the poison, and with a handful of expected flips the sampling could be the
# whole effect.
#
# --max-tokens 16384, NOT 8192. Truncation is arm-dependent: the poisons that
# work also make the model ramble past the cap. At 8192 the qc arm lost 33 of
# 90 turns to truncation against 2 for clean, which is a confound in the
# direction of the result. At 16384 it loses 11 and accuracy is unchanged, so
# the effect is real -- but only the larger cap can show that.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
cd "$HERE"

# Flowcept writes provenance OFFLINE -- no Redis, no Mongo, no service to run.
# Without this it falls back to its packaged defaults and tries to reach a
# message queue that is not there.
export FLOWCEPT_SETTINGS_PATH="${FLOWCEPT_SETTINGS_PATH:-$HERE/configs/flowcept_gpu.yaml}"
# Keeps the engine in-process, so the connector's probe sees the same address
# space as the runner. With multiprocessing on, the capture is silently empty.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

# CAPS CUDA COMPILATION AT WARMUP. THIS IS NOT A TUNING KNOB -- WITHOUT IT THE
# BOX GOES DOWN. FlashInfer JIT-builds the MoE kernel on the first run for a
# given model and compute capability, and ninja defaults to one nvcc per core
# (~20 here). Each is memory-hungry, and on a unified-memory part they compete
# with the model weights already resident in the SAME pool, so the OOM killer
# starts taking processes -- ninja reports "exit status 137" if you are lucky,
# and the machine falls over if you are not.
#
# Measured in exp-2026-10-01: 2 works with a 57 GiB MoE model; 4 untested.
# The cost is bounded -- FlashInfer caches what it builds, so only the first
# run with a given model compiles at all.
export MAX_JOBS="${MAX_JOBS:-2}"

MODEL="${OPAL_MODEL:-Qwen/Qwen3.5-35B-A3B}"
LIMIT="${OPAL_LIMIT:-20}"
# Prefer the project venv, so this works without activating anything.
VENV="$HERE/.venv/bin/python"
PY="${OPAL_PYTHON:-$([[ -x "$VENV" ]] && echo "$VENV" || echo python3)}"

# THE VENV'S bin/ MUST BE ON PATH, not just its python. FlashInfer JIT-compiles
# the MoE kernel during the engine's profile run and shells out to `ninja`,
# which pip installs into .venv/bin. Calling .venv/bin/python by absolute path
# does NOT put that directory on PATH, so the build dies with a bare
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# thrown from subprocess, 70 GiB of weight loading after it could have been
# caught.
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
POISONED="bench/opal/data/poisoned"
mkdir -p runs runs/.preflight "$POISONED"

# Record our process GROUP so stop_run.sh can signal exactly this suite.
# Matching on command lines instead would also match the shell that launched
# it, and killing that group takes your ssh session with it.
echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > runs/.suite.pid
trap 'rm -f runs/.suite.pid' EXIT

# Every arm that has a CSV form. `history` is deliberately absent: it
# substitutes the ASSISTANT's reply, and a CSV holds no assistant turns, so it
# cannot be expressed as a dataset. Run it separately if you need it:
#   python scripts/run_experiment.py --poison configs/poison/history-runtime.json ...
ALL=(clean outfmt rules qc schema null alias_swap floor_raise injection relocate)
ARMS=("$@")
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=("${ALL[@]}")

# CUDA graphs cost minutes of capture at startup and buy decode throughput.
# Worth it for the full suite, wasted on a smoke test -- and the capture phase
# re-touches every weight, so on a box where the checkpoint is larger than RAM
# it page-faults the whole model back off disk and looks like a hang.
GRAPHS=(--cuda-graphs)
[[ "${OPAL_EAGER:-0}" == 1 ]] && GRAPHS=()

# Engine tuning without editing this file. Memory is the usual reason:
# weights are ~67 GiB for Qwen3.5-35B-A3B, and what is left for the KV cache
# depends on the card. Measured on the pinned set, the longest prompt is 5067
# tokens, so with --max-tokens 16384 the true ceiling is ~21.5k and
# --max-model-len 24576 is safe and returns ~25% of the cache.
EXTRA=()
[[ -n "${OPAL_EXTRA_ARGS:-}" ]] && read -r -a EXTRA <<<"$OPAL_EXTRA_ARGS"

COMMON=(--model "$MODEL"
        --enable-thinking
        --max-tokens 16384 --max-model-len 32768
        --temperature 0.0
        --max-num-seqs 8 --limit "$LIMIT"
        "${GRAPHS[@]}" "${EXTRA[@]}")

csv_for() {
  if [[ "$1" == clean ]]; then echo "bench/opal/data/opal_chains.csv"
  else echo "$POISONED/$1.csv"; fi
}

# ---- 1. build the datasets ------------------------------------------------
# Regenerated every time, because token parity is model-dependent: a manifest
# that balanced under one tokenizer can shift every index under another, and a
# stale CSV would do it silently.
echo "=== building poisoned datasets for $MODEL ==="
$PY scripts/make_poison_csv.py --all --model "$MODEL" --limit "$LIMIT" \
    --csv bench/opal/data/opal_chains.csv --out "$POISONED"

# ---- 1b. pin every arm to the chains all of them poison -------------------
PIN=()
CORE_IDS=()      # stays empty under OPAL_PIN=0; suite_arm.txt reads it either way
if [[ "${OPAL_PIN:-1}" != "0" ]]; then
  CORE="$($PY scripts/common_chains.py --oneline)"
  [[ -n "$CORE" ]] || { echo "FATAL: no common poisoned core"; exit 1; }
  # --limit truncates AFTER --chain-ids (chains.select), so a stale --limit
  # would silently cut the pinned set. Pass 0 and let the ids decide.
  read -r -a CORE_IDS <<<"$CORE"
  PIN=(--chain-ids "${CORE_IDS[@]}" --limit 0)
  echo
  echo "=== pinned to ${#CORE_IDS[@]} chains poisoned by every arm ==="
  printf '    %s\n' "${CORE_IDS[@]}"
else
  echo
  echo "=== OPAL_PIN=0: running the full set; arms poison different subsets ==="
fi

# ---- 2. pre-flight every arm before any GPU work --------------------------
echo
# Tools the engine shells out to at RUNTIME. Checked here because each one
# surfaces hundreds of frames deep, after the model is loaded, naming neither
# the tool nor the package that provides it.
for tool in ninja; do
  command -v "$tool" >/dev/null || {
    echo "FATAL: '$tool' is not on PATH."
    echo "  FlashInfer shells out to it to JIT the MoE kernel during warmup,"
    echo "  and fails with a bare FileNotFoundError after loading the model."
    echo "  Install it into the venv:  $PY -m pip install $tool"
    echo "  (or re-run ./scripts/setup_box.sh, which does)"
    exit 1
  }
done
echo "runtime tools: $(command -v ninja)   MAX_JOBS=$MAX_JOBS"
if [[ "$MAX_JOBS" -gt 4 ]]; then
  echo "WARNING: MAX_JOBS=$MAX_JOBS. Each nvcc competes with the resident model"
  echo "  weights for the same memory; >4 has taken this class of box down."
fi

# ---- will this fit, and will it leave the box usable? -------------------
#
# gpu_memory_utilization is a fraction of TOTAL device memory, not of what is
# free. On a unified-memory part (GPU and host share one pool) the default
# 0.88 of a 122 GiB box means vLLM reserves ~107 GiB -- out of the same pool
# another user's job is sitting in. That does not merely fail; it can take
# their job down. Checked here, before anything is loaded.
$PY - "$MODEL" "${COMMON[@]}" <<'MEMCHK' || exit 1
import re, shutil, subprocess, sys

util = 0.88                      # run_experiment.py's default
for i, a in enumerate(sys.argv):
    if a == "--gpu-memory-utilization" and i + 1 < len(sys.argv):
        util = float(sys.argv[i + 1])

def nvidia():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        t, u = (float(x) for x in out[0].split(","))
        return t / 1024, u / 1024                       # GiB
    except Exception:
        return None                 # [N/A] on unified parts, or no driver

def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = float(v.strip().split()[0]) / 1048576    # GiB
    return d["MemTotal"], d["MemAvailable"]

gpu = nvidia()
host_total, host_avail = meminfo()
if gpu and gpu[0] < host_total * 0.9:
    total, avail, pool = gpu[0], gpu[0] - gpu[1], "GPU (discrete)"
else:
    # nvidia-smi reports [N/A] on unified-memory parts, and when it does report
    # it is the same pool as host RAM. Either way, host availability is binding.
    total, avail, pool = host_total, host_avail, "unified (GPU + host share it)"

want = total * util
print(f"  memory pool: {pool}")
print(f"  total {total:.1f} GiB, available {avail:.1f} GiB, "
      f"--gpu-memory-utilization {util} would reserve {want:.1f} GiB")
if want > avail:
    print(f"""
FATAL: vLLM would reserve {want:.1f} GiB but only {avail:.1f} GiB is free.
  On a shared pool this does not just fail -- it can take another user's job
  with it. {total - avail:.1f} GiB is already in use by something else.

  Either wait for the pool, or cap the reservation to fit. A safe value here:
      OPAL_EXTRA_ARGS='--gpu-memory-utilization {max(0.05, (avail - 2) / total):.2f}' ./scripts/run_suite.sh
  which reserves {max(0.05, (avail - 2) / total) * total:.1f} GiB and leaves 2 GiB of margin.
  Your model's weights must still fit inside that.""")
    sys.exit(1)
MEMCHK

echo "=== pre-flight (no GPU) ==="
for arm in "${ARMS[@]}"; do
  csv="$(csv_for "$arm")"
  [[ -f "$csv" ]] || { echo "FATAL: $arm has no dataset at $csv"; exit 1; }
  $PY scripts/run_experiment.py --dry-run --csv "$csv" "${COMMON[@]}" "${PIN[@]}" \
      --out-dir "runs/.preflight/$arm" >"runs/.preflight/$arm.log" 2>&1 \
    || { echo "FATAL: pre-flight failed for $arm"; tail -20 "runs/.preflight/$arm.log"; exit 1; }
  echo "  ok   $arm  ($csv)"
done
[[ -n "${DRY:-}" ]] && { echo; echo "DRY=1: stopping before the GPU."; exit 0; }

# ---- 3. run them, in sequence, on the card --------------------------------
echo
OK=0
for arm in "${ARMS[@]}"; do
  out="runs/${STAMP}-${arm}"
  # EVERYTHING FOR A RUN LIVES IN THAT RUN'S DIRECTORY, the log included. A
  # log beside the directory is the one artifact that gets separated from its
  # run when the results are moved, pulled or archived -- and it is the one
  # you need when the run failed.
  mkdir -p "$out"
  echo "=== $arm -> $out ==="
  # THE MANIFEST IS RESULTS METADATA, SO IT TRAVELS WITH THE RUN. It records
  # exactly which chains this arm poisoned and what it substituted -- the
  # analysis is meaningless without it. But it lives in a directory this
  # script REGENERATES on every invocation, so a later smoke test at a
  # smaller --limit silently rewrites it to describe fewer chains, and the
  # analysis then quietly covers one chain per arm instead of ten. Copying it
  # in makes the run self-describing and immune to that.
  [[ "$arm" == clean ]] || cp "$POISONED/$arm.manifest.json" "$out/poison_manifest.json"
  { echo "arm=$arm"; echo "suite_stamp=$STAMP"; echo "model=$MODEL"
    echo "csv=$(csv_for "$arm")"; echo "host=$(hostname)"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "pinned_chains=${#CORE_IDS[@]}"
    echo "args=${COMMON[*]} ${PIN[*]}"; } > "$out/suite_arm.txt"
  $PY scripts/run_experiment.py --csv "$(csv_for "$arm")" "${COMMON[@]}" "${PIN[@]}" \
      --workflow-id "opal-${STAMP}-${arm}" \
      --out-dir "$out" 2>&1 | tee "$out/run.log" || true

  if [[ -s "$out/turns.jsonl" ]]; then
    OK=$((OK + 1))
  else
    echo "FAILED: $arm (no turns written)"
    # A failure before ANY arm has produced turns is systematic -- a missing
    # tool, the model not fitting, a bad flag. Running the other nine would
    # reload 67 GiB of weights nine more times to fail identically.
    if [[ "$OK" -eq 0 ]]; then
      echo
      echo "ABORTING: the first arm produced nothing, so this is a setup"
      echo "problem, not a bad arm. The whole suite would fail the same way."
      echo "The error is at the end of $out/run.log:"
      tail -5 "$out/run.log" | sed 's/^/    /'
      exit 1
    fi
    echo "(continuing; $OK arm(s) already succeeded)"
  fi
done

echo
echo "=== done. Reduce and render with: ==="
echo "    $PY scripts/attn_batch.py"
