#!/bin/bash
# One-time setup on the GPU box. Idempotent: safe to re-run.
#
#   ./scripts/setup_box.sh
#   VLLM_VERSION=0.28.0 ./scripts/setup_box.sh
#
# Creates .venv at the repo root, installs vLLM, then flowcept and vllm-attn-connector
# EDITABLE from source. Both must be built against the vLLM they will run
# under -- the connector is an out-of-tree KV connector and imports vLLM's
# connector base and attention-backend registry directly.
#
# Siblings are reused if they are already on the box (set OPAL_FLOWCEPT_DIR /
# OPAL_CONNECTOR_DIR to point at them); otherwise they are cloned into
# vendor/. A checkout you own is never edited silently -- the two upstream
# fixes below are applied only to clones this script created.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
ROOT="$HERE"
VENV="${OPAL_VENV:-$ROOT/.venv}"
VENDOR="${OPAL_VENDOR_DIR:-$ROOT/vendor}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
FLOWCEPT_REPO="${OPAL_FLOWCEPT_REPO:-https://github.com/spotter-ai-genesis/flowcept.git}"
CONNECTOR_REPO="${OPAL_CONNECTOR_REPO:-https://github.com/GueroudjiAmal/vllm-attn-connector.git}"
# The branch carrying hybrid + multimodal KV support. Qwen3.5 is a hybrid
# model (gated-delta-net + full attention, 10 of 40 layers); without this the
# connector finds no block table for the scored layers and captures ZEROS.
CONNECTOR_BRANCH="${OPAL_CONNECTOR_BRANCH:-hybrid-multimodal-kv-support}"

say() { printf '==> %s\n' "$*" >&2; }

# ---- venv ----------------------------------------------------------------
#
# BUILT ON A uv-MANAGED CPYTHON, NOT THE SYSTEM ONE. The box's python3 failed
# twice over: `python3 -m venv` produced a venv with no pip (Debian splits
# ensurepip into python3-venv), and it has no Python.h (headers live in
# python3-dev). Triton compiles its CUDA shim at RUNTIME and needs those
# headers, so neither is optional and both need root to fix in place.
#
# A uv-managed interpreter ships its headers and needs no ensurepip, so one
# decision removes both failures and needs no sudo. uv is already how this
# project installs on Polaris.
#
# OPAL_SYSTEM_PYTHON=1 forces the old behaviour, for a box whose python3 is
# already complete.
PYVER="${OPAL_PYTHON_VERSION:-3.12}"

have_headers() {
    "$1" -c 'import sysconfig,pathlib,sys
p = pathlib.Path(sysconfig.get_paths()["include"])/"Python.h"
sys.exit(0 if p.exists() else 1)' 2>/dev/null
}
have_pip() { "$1" -m pip --version >/dev/null 2>&1; }

if [[ "${OPAL_SYSTEM_PYTHON:-0}" == 1 ]]; then
    [[ -d "$VENV" ]] || { say "creating $VENV (system python3)"; python3 -m venv "$VENV"; }
else
    if ! command -v uv >/dev/null 2>&1; then
        say "installing uv (no sudo needed; set OPAL_SYSTEM_PYTHON=1 to skip)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || {
        echo "!! uv is still not on PATH. Add it and re-run:" >&2
        echo '     export PATH="$HOME/.local/bin:$PATH"' >&2
        exit 1
    }
    # A .venv left from a failed system-python attempt has no pip and no
    # headers; reusing it reproduces the exact failure this replaces.
    if [[ -d "$VENV" ]] && { ! have_pip "$VENV/bin/python" || ! have_headers "$VENV/bin/python"; }; then
        say "discarding $VENV: it has no pip and/or no headers"
        rm -rf "$VENV"
    fi
    if [[ ! -d "$VENV" ]]; then
        say "installing CPython $PYVER via uv (ships its own headers)"
        uv python install "$PYVER"
        say "creating $VENV"
        uv venv --python "$PYVER" "$VENV"
    fi
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"

# One installer for the rest of the script, whichever route got us here.
if command -v uv >/dev/null 2>&1 && [[ "${OPAL_SYSTEM_PYTHON:-0}" != 1 ]]; then
    PIP=(uv pip install --python "$VENV/bin/python")
else
    have_pip "$VENV/bin/python" || {
        echo "!! $VENV has no pip. The system python3 cannot bootstrap one" >&2
        echo "   (Debian ships ensurepip in python3-venv). Either:" >&2
        echo "     sudo apt-get install -y python3-venv python3-dev" >&2
        echo "   or drop OPAL_SYSTEM_PYTHON and let uv provide the interpreter:" >&2
        echo "     rm -rf ${VENV@Q} && ./scripts/setup_box.sh" >&2
        exit 1
    }
    PIP=("$VENV/bin/python" -m pip install)
    "${PIP[@]}" -q --upgrade pip
fi

have_headers "$VENV/bin/python" || {
    cat >&2 <<MSG
!! Python.h is missing from $("$VENV/bin/python" -c 'import sys; print(sys.base_prefix)'),
   so Triton cannot build its CUDA shim.

   You are on the system interpreter. Either install its headers:
       sudo apt-get install -y python3-dev     # or python3-devel
   or let uv provide one that already has them:
       unset OPAL_SYSTEM_PYTHON
       rm -rf ${VENV@Q} && ./scripts/setup_box.sh
MSG
    exit 1
}
say "python $("$VENV/bin/python" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))') with headers and an installer"

# ---- vLLM ----------------------------------------------------------------
say "installing vLLM $VLLM_VERSION (pulls CUDA torch and triton)"
"${PIP[@]}" -q "vllm==$VLLM_VERSION"

# ---- siblings ------------------------------------------------------------
resolve() {  # name repo branch varname
  local name="$1" repo="$2" branch="$3" var="$4" dir="" owned=0
  for cand in "${!var:-}" "$ROOT/../$name" "$VENDOR/$name"; do
    [[ -n "$cand" && -d "$cand/src" ]] && { dir="$cand"; break; }
  done
  if [[ -z "$dir" ]]; then
    dir="$VENDOR/$name"; owned=1
    say "cloning $name -> $dir"
    mkdir -p "$VENDOR"
    git clone --quiet ${branch:+--branch "$branch"} "$repo" "$dir"
  fi
  [[ "$dir" == "$VENDOR/"* ]] && owned=1
  printf '%s %s' "$dir" "$owned"
}

read -r FLOWCEPT_DIR FLOWCEPT_OWNED <<<"$(resolve flowcept "$FLOWCEPT_REPO" "" OPAL_FLOWCEPT_DIR)"
read -r CONNECTOR_DIR CONNECTOR_OWNED <<<"$(resolve vllm-attn-connector "$CONNECTOR_REPO" "$CONNECTOR_BRANCH" OPAL_CONNECTOR_DIR)"

# ---- the connector must actually support hybrid KV ----------------------
#
# CHECKED BY CONTENT, NOT BRANCH NAME. `--branch` above applies only when this
# script CLONES; a checkout already on the box is reused on whatever branch it
# happens to be on, and the branch may also have been merged upstream under a
# different name. So look for the capability itself.
#
# Qwen3.5 is hybrid -- gated-delta-net plus full attention, 10 of 40 layers --
# and is multimodal, so its text config nests under hf_text_config. Without
# this support the connector finds no block table for the scored layers and
# the capture is ALL ZEROS, with no error anywhere: valid records, correct
# shapes, every number 0.0. That has already cost five runs, so it is fatal
# here rather than a warning.
check_connector() {
  local f="$1/src/vllm_attn_connector/connector.py"
  [[ -f "$f" ]] || { echo "!! $f not found; is $1 a vllm-attn-connector checkout?" >&2; return 1; }
  local missing=()
  grep -q 'hf_text_config' "$f" || missing+=("multimodal text-config lookup (hf_text_config)")
  grep -q '_scored' "$f"        || missing+=("per-KV-group layer scoring (_scored)")
  [[ ${#missing[@]} -eq 0 ]] && return 0
  cat >&2 <<MSG
!! $1
   does not support hybrid KV caches. Missing:
$(printf '     - %s\n' "${missing[@]}")
   on branch $(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') @ $(git -C "$1" rev-parse --short HEAD 2>/dev/null || echo '?')

   Qwen3.5 would capture ALL ZEROS through it, silently -- valid records,
   correct shapes, every value 0.0. Fix the checkout:

       git -C ${1@Q} fetch --all
       git -C ${1@Q} checkout ${CONNECTOR_BRANCH@Q}

   or let this script use its own clone instead:

       OPAL_CONNECTOR_DIR= OPAL_VENDOR_DIR=${VENDOR@Q} ./scripts/setup_box.sh

   or apply vendor-patches/vllm-attn-connector-hybrid-kv.patch to it.
MSG
  return 1
}
check_connector "$CONNECTOR_DIR" || exit 1
say "connector supports hybrid KV ($(git -C "$CONNECTOR_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"

# ---- two upstream fixes, applied only to clones we own -------------------
#
# 1. PEP 639. setuptools >= 77 refuses a pyproject carrying BOTH a license
#    expression and a legacy `License ::` classifier, so the editable install
#    fails on any current box.
fix_pep639() {
  local py="$1/pyproject.toml" owned="$2"
  [[ -f "$py" ]] || return 0
  grep -q 'License :: OSI Approved' "$py" || return 0
  grep -qE '^license *= *"' "$py" || return 0
  if [[ "$owned" == 1 ]]; then
    say "$(basename "$1"): dropping the legacy License classifier (PEP 639)"
    sed -i '/License :: OSI Approved/d' "$py"
  else
    echo "!! $py has both a license expression and a 'License ::' classifier;"
    echo "   setuptools >= 77 rejects that and the install below will fail."
    echo "   This checkout is yours -- fix it there:  sed -i '/License :: OSI Approved/d' $py"
  fi
}
# 2. Flowcept calls nvmlDeviceGetMemoryInfo() unconditionally, before checking
#    whether "used" was requested. On a unified-memory part (GB10 / DGX Spark,
#    GH200) there is no discrete framebuffer, so it raises NotSupported -- once
#    per captured task -- and takes temperature, power, name and id down with
#    it even though all four work there.
fix_nvml() {
  local f="$1/src/flowcept/flowceptor/telemetry_capture.py" owned="$2"
  [[ -f "$f" ]] || return 0
  grep -q 'nvmlDeviceGetMemoryInfo(device)$' "$f" || grep -q 'conf_gpu_telemetry' "$f" || return 0
  if [[ "$owned" != 1 ]]; then
    echo "!! $f calls nvmlDeviceGetMemoryInfo() unconditionally, which fails on" >&2
    echo "   unified-memory GPUs and disables ALL GPU telemetry there." >&2
    echo "   This checkout is yours; patch and commit it if you want the fix." >&2
    return 0
  fi
  "$VENV/bin/python" - "$f" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()

# __get_gpu_info_nvidia is a @staticmethod taking (gpu_conf, gpu_ix). It has
# NO self -- an earlier version of this patch used one and every capture
# raised NameError, taking all GPU telemetry down with it. Guard on gpu_conf.
GOOD = ('        nvidia_info = nvmlDeviceGetMemoryInfo(device) '
        'if "used" in gpu_conf else None\n')
ORIG = '        nvidia_info = nvmlDeviceGetMemoryInfo(device)\n'
BROKEN = ("        nvidia_info = (nvmlDeviceGetMemoryInfo(device)\n"
          "                       if 'used' in self.conf_gpu_telemetry else None)\n")

if GOOD in s:
    print("   flowcept: NVML probe already lazy")
elif BROKEN in s:
    p.write_text(s.replace(BROKEN, GOOD, 1))
    print("   flowcept: REPAIRED a bad NVML patch (it referenced self)")
elif ORIG in s:
    p.write_text(s.replace(ORIG, GOOD, 1))
    print("   flowcept: NVML memory probe made lazy")
else:
    sys.exit("!! flowcept telemetry_capture.py does not match any known form; "
             "leaving it alone. GPU telemetry may error once per task.")
PY
}
fix_pep639 "$FLOWCEPT_DIR"  "$FLOWCEPT_OWNED"
fix_nvml   "$FLOWCEPT_DIR"  "$FLOWCEPT_OWNED"
fix_pep639 "$CONNECTOR_DIR" "$CONNECTOR_OWNED"

say "installing flowcept + connector (editable)"
"${PIP[@]}" -q -e "${FLOWCEPT_DIR}[telemetry,nvidia]"
"${PIP[@]}" -q -e "$CONNECTOR_DIR"
# ninja is NOT optional: FlashInfer JIT-compiles kernels during vLLM warmup and
# shells out to `ninja`. Without it the engine dies with a bare FileNotFoundError
# nested inside "Engine core initialization failed", naming neither.
"${PIP[@]}" -q -r "$ROOT/requirements-local.txt" ninja "huggingface_hub[cli]"

uv pip freeze --python "$VENV/bin/python" > "$ROOT/requirements.gpu.lock" 2>/dev/null \
  || "$VENV/bin/python" -m pip freeze > "$ROOT/requirements.gpu.lock"

# ---- verify --------------------------------------------------------------
# ninja lands in the venv's bin/, which is on PATH only when the venv is
# ACTIVATED. The scripts here call .venv/bin/python directly instead, so check
# the binary is reachable the way a run will actually look for it.
if ! PATH="$VENV/bin:$PATH" command -v ninja >/dev/null 2>&1; then
    echo "!! ninja is not on PATH even with $VENV/bin prepended." >&2
    echo "   FlashInfer cannot JIT the MoE kernel without it. Install it:" >&2
    echo "     $VENV/bin/python -m pip install ninja" >&2
    echo "   or system-wide:  sudo apt-get install -y ninja-build" >&2
    exit 1
fi
say "ninja: $(PATH="$VENV/bin:$PATH" command -v ninja)"

say "verifying"
python - <<'PY'
import importlib.metadata as md
import vllm_attn_connector
print(f"  vllm      {md.version('vllm')}")
print(f"  flowcept  {md.version('flowcept')}")
ok = vllm_attn_connector.install_probe()
print(f"  probe installs: {ok}")
raise SystemExit(0 if ok else 1)
PY

cat <<MSG

setup complete.
  venv       $VENV
  flowcept   $FLOWCEPT_DIR
  connector  $CONNECTOR_DIR ($(git -C "$CONNECTOR_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))
  lockfile   $ROOT/requirements.gpu.lock

Next:  DRY=1 ./scripts/run_suite.sh        # pre-flight, no GPU
       ./scripts/run_suite.sh              # the real thing
MSG
