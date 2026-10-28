#!/bin/bash
# Move this experiment to and from the GPU box over ssh. No git involved.
#
#   export OPAL_BOX=agueroudji@gpuhost
#   export OPAL_BOX_PATH=genesis/experiments/experiments_10_26_26   # the default
#
#   ./scripts/sync_box.sh push             # code + the clean CSV, to the box
#   ./scripts/sync_box.sh push -n          # dry run: what WOULD be sent
#   ./scripts/sync_box.sh pull             # EVERYTHING: runs, tensors, analytics
#   ./scripts/sync_box.sh pull --light     # skip the tensors (attention/, raw/)
#   ./scripts/sync_box.sh pull 20261026T120000Z-qc              # one run
#   ./scripts/sync_box.sh ls               # what runs exist on the box
#
# PUSH IS AN EXPLICIT INCLUDE-LIST, not an exclude-list. This directory
# accumulates venvs, vendored clones, generated datasets and run outputs, and
# with an exclude-list each new one ships by default until someone notices.
# bench/opal/data/poisoned/ is deliberately NOT sent: run_suite.sh regenerates
# it on the box, because token parity is tokenizer-dependent and a stale CSV is wrong
# silently.
#
# PULL BRINGS BACK EVERYTHING BY DEFAULT -- whole run directories including
# the attention tensors, plus analytics-out/ and the poison manifests. A suite
# is 10-20 GB, so this is not cheap, but a partial copy is worse: the thing
# left behind is always the thing an analysis later needs, and by then the box
# may have been reused. --light skips attention/ and raw/ when you only want
# to read results, which is a few MB.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
cd "$HERE"

BOX="${OPAL_BOX:-}"
BOX_PATH="${OPAL_BOX_PATH:-genesis/experiments/experiments_10_26_26}"
SSH_OPTS="${OPAL_SSH_OPTS:-}"
REMOTE="$BOX_PATH"

usage() { sed -n '2,28p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }
[[ $# -gt 0 ]] || usage 2
CMD="$1"; shift
[[ "$CMD" == help || "$CMD" == -h || "$CMD" == --help ]] && usage 0

if [[ -z "$BOX" ]]; then
    echo "!! set OPAL_BOX to the machine, e.g.  export OPAL_BOX=agueroudji@gpuhost" >&2
    exit 2
fi
# OPAL_BOX is an ssh destination, not a URL. Left alone, a browser-shaped
# address surfaces three layers down as "Could not resolve hostname http",
# because rsync reads everything before the first ':' as the host.
BOX="${BOX#http://}"; BOX="${BOX#https://}"; BOX="${BOX#ssh://}"; BOX="${BOX%/}"
if [[ "$BOX" == */* || "$BOX" == *:* ]]; then
    echo "!! OPAL_BOX=\"${OPAL_BOX}\" is not an ssh destination." >&2
    echo "   It must be  user@host  -- no scheme, no path, no colon." >&2
    echo "   The path goes in OPAL_BOX_PATH instead." >&2
    exit 2
fi

# shellcheck disable=SC2086
rsh() { ssh $SSH_OPTS "$BOX" "$@"; }
RS=(rsync -azh --partial)
[[ -n "$SSH_OPTS" ]] && RS+=(-e "ssh $SSH_OPTS")

case "$CMD" in

push)
    PAYLOAD=(scripts configs src tests vendor-patches
             bench/opal/analytics bench/opal/data/opal_chains.csv
             requirements-local.txt requirements-gpu.txt
             pyproject.toml README.md .gitignore)
    for p in "${PAYLOAD[@]}"; do
        [[ -e "$p" ]] || { echo "!! payload item missing locally: $p" >&2; exit 1; }
    done
    echo "==> $BOX:$REMOTE"
    # shellcheck disable=SC2086
    if ! "${RS[@]}" --relative --mkpath "$@" \
        --exclude='__pycache__/' --exclude='*.pyc' \
        --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
        --exclude='out/' \
        "${PAYLOAD[@]}" "$BOX:$REMOTE/"; then
        echo >&2
        echo "!! push failed. The usual causes, in order:" >&2
        echo "   - $BOX_PATH is not writable by you on that host." >&2
        echo "   - no ssh key, so every call prompts. Fix it once:" >&2
        echo "       ssh-copy-id $BOX" >&2
        echo "     or reuse one connection:" >&2
        echo "       export OPAL_SSH_OPTS='-o ControlMaster=auto -o ControlPath=/tmp/opal-%r@%h:%p -o ControlPersist=4h'" >&2
        exit 1
    fi
    # VERIFY, then chmod. rsync preserves the mode, but a payload item that
    # lost +x locally (a fresh clone, a copy through a filesystem without it)
    # arrives unrunnable and fails with "Permission denied" naming no cause.
    #
    # `find -exec` rather than a glob: a glob in an ssh command line is
    # expanded by the LOCAL shell first, and when it matches nothing locally
    # the unexpanded pattern is sent on -- which is how this reported
    # "cannot access .../scripts/*.sh" while the push itself had succeeded.
    N=$(rsh "find ${REMOTE@Q}/scripts -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l")
    if [[ "${N:-0}" -eq 0 ]]; then
        echo >&2
        echo "!! push reported success but $BOX:$REMOTE/scripts has no .sh files." >&2
        echo "   What is actually at the destination:" >&2
        rsh "ls -la ${REMOTE@Q} 2>&1 | head -20 | sed 's|^|       |'" >&2
        echo "   Check OPAL_BOX_PATH=${BOX_PATH}" >&2
        exit 1
    fi
    rsh "find ${REMOTE@Q}/scripts -maxdepth 1 -name '*.sh' -exec chmod +x {} +"
    echo "verified: $N scripts at $BOX:$REMOTE/scripts"
    echo
    echo "pushed. On the box:"
    echo "    cd $BOX_PATH && ./scripts/setup_box.sh && DRY=1 ./scripts/run_suite.sh"
    ;;

ls)
    rsh "ls -1t ${REMOTE@Q}/runs 2>/dev/null" \
      || { echo "!! no runs/ at $BOX:$REMOTE" >&2; exit 1; }
    ;;

pull)
    LIGHT=0
    [[ "${1:-}" == "--light" ]] && { LIGHT=1; shift; }
    [[ "${1:-}" == "--attention" ]] && shift        # old spelling, now the default
    WHAT="${1:-}"

    # Check the remote path BEFORE rsync, whose own message names a path
    # nobody chose ("change_dir ... failed") and never mentions the cause:
    # OPAL_BOX_PATH unset in THIS shell, so the default was used. The variable
    # does not survive opening a new terminal, which is exactly when this bites.
    if ! rsh "test -d ${REMOTE@Q}" 2>/dev/null; then
        echo "!! $BOX:$REMOTE does not exist." >&2
        if [[ -z "${OPAL_BOX_PATH:-}" ]]; then
            echo "   OPAL_BOX_PATH is UNSET here, so pull used the default:" >&2
            echo "       $BOX_PATH" >&2
            echo "   Set it (and put it in ~/.bashrc) if the box differs." >&2
        else
            echo "   OPAL_BOX_PATH=$OPAL_BOX_PATH -- wrong, or nothing pushed yet." >&2
        fi
        exit 1
    fi
    if ! rsh "test -d ${REMOTE@Q}/runs" 2>/dev/null; then
        echo "!! no runs/ on the box yet. runs/ is created by the first run," >&2
        echo "   not by setup. Start one:  ./scripts/run_suite.sh" >&2
        exit 1
    fi
    # A named run that is not there deserves the list, not a transfer of
    # nothing: rsync exits 0 having copied no files when the glob matches none,
    # which reads as success.
    if [[ -n "$WHAT" ]] && ! rsh "test -e ${REMOTE@Q}/runs/${WHAT@Q}" 2>/dev/null; then
        echo "!! no run named '$WHAT' on the box. What is there, newest first:" >&2
        rsh "ls -1t ${REMOTE@Q}/runs | sed 's|^|       |'" >&2
        exit 1
    fi

    mkdir -p runs analytics-out
    EX=()
    if [[ "$LIGHT" == 1 ]]; then
        EX=(--exclude='attention/' --exclude='raw/')
        echo "    (--light: tensors excluded)"
    else
        REMOTE_SZ=$(rsh "du -sh ${REMOTE@Q}/runs 2>/dev/null | cut -f1" || echo '?')
        echo "    pulling EVERYTHING, including tensors (remote runs/ is $REMOTE_SZ)."
        echo "    Pass --light to skip attention/ and raw/."
    fi
    echo "==> runs  <- $BOX:$REMOTE/runs/${WHAT}"
    "${RS[@]}" --info=progress2 "${EX[@]}" \
        "$BOX:$REMOTE/runs/${WHAT:+$WHAT}" runs/

    # The manifests say which chains each arm poisoned and what it changed.
    # They are regenerated by every run_suite.sh invocation, so the copies on
    # the box are the ones that matter -- and only until the next run.
    if rsh "ls ${REMOTE@Q}/bench/opal/data/poisoned/*.manifest.json" >/dev/null 2>&1; then
        echo "==> poison manifests  <- $BOX"
        mkdir -p bench/opal/data/poisoned
        "${RS[@]}" "$BOX:$REMOTE/bench/opal/data/poisoned/*.manifest.json" \
            bench/opal/data/poisoned/
    fi

    if rsh "test -d ${REMOTE@Q}/analytics-out" 2>/dev/null; then
        echo "==> analytics-out  <- $BOX"
        "${RS[@]}" --info=progress2 "$BOX:$REMOTE/analytics-out/" analytics-out/
    else
        echo "note: no analytics-out on the box. Reduce there first, it is far" >&2
        echo "      cheaper than pulling tensors:  python scripts/attn_batch.py" >&2
    fi
    echo
    echo "pulled into runs/ and analytics-out/"
    ;;

*)
    echo "!! unknown command: $CMD" >&2
    usage 2
    ;;
esac
