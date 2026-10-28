#!/bin/bash
# Stop a running suite and release the GPU.
#
#   ./scripts/stop_run.sh          # show what is running, stop nothing
#   ./scripts/stop_run.sh --kill   # stop it
#
# WHY NOT CTRL-C. During CUDA-graph capture the thread is inside a CUDA call,
# and when the checkpoint is larger than RAM it page-faults in uninterruptible
# (D) state, where even SIGKILL waits for the syscall to return. Ctrl-C also
# only reaches the foreground job.
#
# WHY A PIDFILE, NOT A COMMAND-LINE MATCH. run_suite.sh runs the arms in a
# loop, so killing just the python ends one arm and the loop starts the next.
# The group has to go. But `ps | grep run_experiment.py` also matches the
# SHELL that launched it -- an ssh session, an editor, another grep -- and
# killing that group logs you out. An earlier version of this script did
# exactly that. So the suite records its own process group and we signal that.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$HERE/runs/.suite.pid"

gpu_apps() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
        | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"
}

if [[ ! -f "$PIDFILE" ]]; then
    echo "no suite pidfile at $PIDFILE -- nothing started by run_suite.sh is running."
    echo
    echo "If a run is still holding the GPU (started by hand, or the suite was"
    echo "killed leaving a child), find it here and kill that PID directly:"
    gpu_apps
    exit 0
fi

read -r SPID SPGID < "$PIDFILE"
if ! kill -0 "$SPID" 2>/dev/null; then
    echo "stale pidfile: suite pid $SPID is gone. Removing it."
    rm -f "$PIDFILE"
    echo "GPU compute apps (kill any PID here by hand if memory is still held):"
    gpu_apps
    exit 0
fi

echo "suite pid $SPID, process group $SPGID"
ps -eo pid,pgid,stat,etime,comm | awk -v g="$SPGID" 'NR==1 || $2==g' | sed 's/^/  /'

if ps -eo pgid,stat | awk -v g="$SPGID" '$1==g{print $2}' | grep -q '^D'; then
    echo
    echo "note: a process is in D (uninterruptible I/O) state. It will not die"
    echo "      until that read completes -- usually the checkpoint paging off"
    echo "      disk. Expect a delay even with SIGKILL."
fi

if [[ "${1:-}" != "--kill" ]]; then
    echo
    echo "re-run with --kill to stop it."
    exit 0
fi

echo "==> TERM process group $SPGID"
kill -TERM -- "-$SPGID" 2>/dev/null
for _ in $(seq 15); do
    sleep 1
    kill -0 "$SPID" 2>/dev/null || break
done
if kill -0 "$SPID" 2>/dev/null; then
    echo "==> KILL process group $SPGID"
    kill -KILL -- "-$SPGID" 2>/dev/null
    sleep 2
fi
kill -0 "$SPID" 2>/dev/null && echo "still alive (likely D state); re-run shortly." \
                            || { echo "stopped."; rm -f "$PIDFILE"; }

echo
echo "GPU compute apps still holding memory:"
gpu_apps
echo
echo "A killed run leaves a partial directory with no summary.json, so"
echo "scripts/attn_batch.py lists it as skipped rather than reducing it:"
ls -1dt "$HERE"/runs/*/ 2>/dev/null | head -3 | sed 's/^/  /'
