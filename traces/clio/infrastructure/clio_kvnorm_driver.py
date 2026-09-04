#!/usr/bin/env python3
"""Multi-prompt clio-agent driver for the fused clio+kvnorm Flowcept dataset.

Sends several varied prompts through the clio GACT server against a local vLLM
OpenAI-compatible server.  Each prompt becomes at least one LM call (more if the
agent does additional reasoning steps), which triggers:

  - clio Flowcept provenance: workflow / task / lm.* records emitted via
    FlowceptProvenanceProvider → Redis MQ → Mongo
  - kvnorm Flowcept provenance: kv_token_importance tasks emitted by the vLLM
    connector on each completed request → same Redis MQ → same Mongo

Run from the clio-agent venv with env vars already set:
    CLIO_LM_PROVIDER=vllm
    CLIO_LM_API_BASE=http://127.0.0.1:8000/v1
    CLIO_LM_MODEL=granite-4.2-30b
    CLIO_LM_API_KEY=EMPTY
    CLIO_LM_MAX_TOKENS=1024
    CLIO_LM_TEMPERATURE=0.0
    CLIO_ARC_STORE=local
    CLIO_LIVE_STREAMING=0
    CLIO_PROVENANCE_PROVIDERS=flowcept
    FLOWCEPT_SETTINGS_PATH=<settings yaml>
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Parse args before clio imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="multi-prompt clio+kvnorm driver")
parser.add_argument("--gact-port", type=int, default=17800)
parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
parser.add_argument("--model", default="granite-4.2-30b")
parser.add_argument("--workspace-root", default="/tmp/clio_kvnorm_ws")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Set CLIO_LM_* env vars BEFORE any clio import (must be in env before import)
# ---------------------------------------------------------------------------
user_dir = os.environ.get("CLIO_USER_DIR", "/tmp/clio_kvnorm_home")
os.makedirs(user_dir, exist_ok=True)

os.environ.setdefault("CLIO_USER_DIR", user_dir)
os.environ.setdefault("CLIO_LM_PROVIDER", "vllm")
os.environ.setdefault("CLIO_LM_API_BASE", args.api_base)
os.environ.setdefault("CLIO_LM_MODEL", args.model)
os.environ.setdefault("CLIO_LM_API_KEY", "EMPTY")
os.environ.setdefault("CLIO_LM_MAX_TOKENS", "1024")
os.environ.setdefault("CLIO_LM_TEMPERATURE", "0.0")
os.environ.setdefault("CLIO_ARC_STORE", "local")
os.environ.setdefault("CLIO_LIVE_STREAMING", "0")
# These two should already be set by the caller:
#   CLIO_PROVENANCE_PROVIDERS=flowcept
#   FLOWCEPT_SETTINGS_PATH=<path>

print(f"[driver] CLIO_USER_DIR          = {os.environ.get('CLIO_USER_DIR')}", flush=True)
print(f"[driver] CLIO_LM_PROVIDER       = {os.environ.get('CLIO_LM_PROVIDER')}", flush=True)
print(f"[driver] CLIO_LM_API_BASE       = {os.environ.get('CLIO_LM_API_BASE')}", flush=True)
print(f"[driver] CLIO_LM_MODEL          = {os.environ.get('CLIO_LM_MODEL')}", flush=True)
print(f"[driver] CLIO_PROVENANCE_PROVIDERS = {os.environ.get('CLIO_PROVENANCE_PROVIDERS', '<unset>')}", flush=True)
print(f"[driver] FLOWCEPT_SETTINGS_PATH = {os.environ.get('FLOWCEPT_SETTINGS_PATH', '<unset>')}", flush=True)
print(f"[driver] GACT port              = {args.gact_port}", flush=True)

import requests  # noqa: E402

gact_base = f"http://127.0.0.1:{args.gact_port}"

# Prompts: varied, including one that encourages multi-step reasoning.
PROMPTS = [
    "What is 17 multiplied by 23? Show your reasoning step by step.",
    "Explain the difference between a stack and a queue data structure in one paragraph.",
    "What are the three laws of thermodynamics? Give a one-sentence summary of each.",
    "If a train travels at 60 mph for 2.5 hours, how far does it go? Show the calculation.",
    "Write a haiku about machine learning.",
]


def _call(method: str, path: str, body=None, params=None, ok=(200, 201)) -> dict:
    r = requests.request(method, f"{gact_base}{path}", json=body, params=params, timeout=300)
    if r.status_code not in ok:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:1200]}")
    return r.json() if r.content else {}


def _wait_health(timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{gact_base}/v1/health", timeout=5)
            if r.status_code in (200, 503):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _wait_lm_ready(timeout: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            info = _call("GET", "/v1/providers/lm/wait", params={"timeout": 20}, ok=(200, 503))
            state = str(info.get("state") or "")
            print(f"[driver]   LM state = {state}", flush=True)
            if state == "ready":
                return True
            if state == "error":
                print(f"[driver] ERROR LM provider: {info}", file=sys.stderr, flush=True)
                return False
        except Exception as exc:
            print(f"[driver]   LM wait error: {exc}", flush=True)
        time.sleep(5)
    return False


def _wait_turn(wsid: str, sid: str, timeout: float = 600.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rows = _call("GET", f"/v1/sessions?workspace_id={wsid}").get("sessions", [])
            row = next((r for r in rows if r.get("id") == sid), {})
            status = str(row.get("status") or "?")
            if status in ("idle", "completed", "error", "waiting_user"):
                return status
        except Exception as exc:
            print(f"[wait_turn] poll error: {exc}", flush=True)
        time.sleep(3)
    return "timed_out"


def run_prompt(wsid: str, prompt: str, turn_num: int) -> str:
    """Create a session, post one prompt, wait for completion, return status."""
    sess = _call("POST", "/v1/sessions", {"title": f"kvnorm-turn-{turn_num}", "workspace_id": wsid})
    sid = str(sess["id"])
    print(f"[driver]   session_id = {sid}  prompt = {prompt[:60]!r}", flush=True)

    _call("POST", f"/v1/sessions/{sid}/messages", {"text": prompt}, ok=(200, 201, 202))
    print(f"[driver]   message posted, waiting...", flush=True)

    final_status = _wait_turn(wsid, sid, timeout=300)
    print(f"[driver]   turn {turn_num} status = {final_status}", flush=True)

    # Show messages (trimmed)
    try:
        msgs = _call("GET", f"/v1/sessions/{sid}/messages").get("messages", [])
        for msg in msgs:
            role = msg.get("role", "?")
            content = msg.get("content") or msg.get("text") or ""
            if isinstance(content, list):
                content = " ".join(
                    (c.get("text") or c.get("content") or str(c))
                    for c in content if isinstance(c, dict)
                )
            print(f"    [{role}] {str(content)[:200]}", flush=True)
    except Exception as exc:
        print(f"    (message retrieval error: {exc})", flush=True)

    return final_status


def main() -> int:
    # Boot GACT server in background thread
    from clio_agent.gact.app import run_server  # type: ignore

    print("[driver] Starting GACT server...", flush=True)
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": args.gact_port},
        daemon=True,
    )
    server_thread.start()

    print("[driver] Waiting for GACT /v1/health...", flush=True)
    if not _wait_health(timeout=90):
        print("[driver] ERROR: GACT server did not come up", file=sys.stderr, flush=True)
        return 1
    print("[driver] GACT server healthy.", flush=True)

    # Bind vLLM provider
    print("[driver] Binding vLLM provider...", flush=True)
    try:
        _call("PUT", "/v1/providers/lm", {
            "provider": "vllm",
            "provider_id": "vllm",
            "api_base": args.api_base,
            "model": args.model,
            "api_key": "EMPTY",
            "temperature": 0.0,
            "max_tokens": 1024,
        })
        print("[driver] PUT /v1/providers/lm OK", flush=True)
    except RuntimeError as exc:
        print(f"[driver] PUT failed (non-fatal, will still wait): {exc}", flush=True)

    if not _wait_lm_ready(timeout=180):
        print("[driver] ERROR: LM provider did not become ready", file=sys.stderr, flush=True)
        return 1
    print("[driver] LM provider READY.", flush=True)

    # Check provider info
    try:
        info = _call("GET", "/v1/providers/lm", ok=(200,))
        print(f"[driver] Provider: provider={info.get('provider')} model={info.get('model')} state={info.get('state')}", flush=True)
    except Exception as exc:
        print(f"[driver] provider info error (non-fatal): {exc}", flush=True)

    # Set allow-all policy
    try:
        _call("PUT", "/v1/policies", {
            "policies": [{"scope": "workspace", "action": "allow", "tool_name_pattern": "*"}]
        })
        print("[driver] allow-all policy set.", flush=True)
    except Exception as exc:
        print(f"[driver] policy set warning (non-fatal): {exc}", flush=True)

    # Create workspace (shared across all prompt sessions)
    os.makedirs(args.workspace_root, exist_ok=True)
    ws = _call("POST", "/v1/workspaces", {"name": "kvnorm-dataset", "root_path": args.workspace_root})
    wsid = str(ws.get("id") or ws.get("workspace_id") or "")
    print(f"[driver] workspace_id = {wsid}", flush=True)

    # Run each prompt as a separate session
    results: list[str] = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n[driver] === Prompt {i}/{len(PROMPTS)} ===", flush=True)
        status = run_prompt(wsid, prompt, i)
        results.append(status)

    # Summary
    print(f"\n[driver] === Summary ({len(results)} turns) ===", flush=True)
    n_ok = sum(1 for s in results if s in ("idle", "completed"))
    n_err = sum(1 for s in results if s == "error")
    n_other = len(results) - n_ok - n_err
    for i, (status, prompt) in enumerate(zip(results, PROMPTS), start=1):
        mark = "OK" if status in ("idle", "completed") else "ERR" if status == "error" else status
        print(f"  [{mark}] turn {i}: {prompt[:50]!r}", flush=True)
    print(f"[driver] turns completed={n_ok} error={n_err} other={n_other}", flush=True)
    print("\n[driver] Done. (Note: clio errors are OK — LM calls and kvnorm records still fire)", flush=True)

    # Allow a moment for Flowcept to flush any pending records
    print("[driver] Sleeping 5s to let Flowcept flush remaining records...", flush=True)
    time.sleep(5)

    return 0  # We always succeed — the dataset is the goal, not clio turn completion


if __name__ == "__main__":
    sys.exit(main())
