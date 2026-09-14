#!/usr/bin/env bash
#SBATCH --job-name=spotter-fused
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#
# SPOTTER-AI: Fused clio+kvnorm provenance dataset in ONE MongoDB.
#
# Architecture:
#   vLLM serve (granite-4.2-30b, +kvnorm connector) ---> Redis MQ ---> MongoDB
#   clio-agent (CLIO_PROVENANCE_PROVIDERS=flowcept)  ---> Redis MQ ---> MongoDB
#
# Both producers use the SAME Redis MQ and the SAME MongoDB (same settings file).
# A single Flowcept DocumentInserter drains the shared MQ into Mongo.
# After the run: mongodump archive + mongoexport JSON + README exported to scratch.
#
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SIF="$SCRATCH/vllm-kvnorm.sif"
REDIS_SIF="$SCRATCH/redis.sif"
MONGO_SIF="$SCRATCH/mongo.sif"
HF_HOME="$SCRATCH/hf"
CH="$SCRATCH/run"
SETTINGS="$SCRATCH/flowcept_pipeline_settings.yaml"
KVSRC="/u/jcernuda/spotterAI/vllm-kvnorm/src"
PKG="$SCRATCH/pipeline_pkgs"         # pandas/pytz/tzdata for Flowcept Mongo DAO

MONGO_DATA="/dev/shm/$USER/mongo_data"
CLIO_REPO="/u/jcernuda/spotterAI/clio-agent"
CLIO_VENV="$CLIO_REPO/.venv"
CLIO_USER_DIR="$SCRATCH/clio_user_dir"
WS_ROOT="$SCRATCH/clio_kvnorm_ws"
LOGDIR="$SCRATCH/spotter_fused_logs_${SLURM_JOB_ID}"
EXPORT_DIR="$SCRATCH/spotter_dataset_${SLURM_JOB_ID}"

MODEL_HF_ID="ibm-granite/granite-4.2-30b"
SERVED_MODEL_NAME="granite-4.2-30b"
VLLM_PORT=8000
GACT_PORT=17800

mkdir -p \
  "$CH/home" "$CH/triton" "$CH/inductor" "$CH/xdg" \
  "$SCRATCH/vllm_cache" \
  "$HF_HOME" \
  "$MONGO_DATA" \
  "/dev/shm/$USER" \
  "$CLIO_USER_DIR" \
  "$WS_ROOT" \
  "$LOGDIR" \
  "$EXPORT_DIR"

echo "== node =="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

# --------------------------------------------------------------------------
# Cleanup trap — runs on EXIT
# --------------------------------------------------------------------------
REDIS_PID=""
MONGO_PID=""
VLLM_PID=""
INSERTER_PID=""

cleanup() {
  echo
  echo "== teardown =="
  [ -n "$VLLM_PID" ]    && kill "$VLLM_PID"    2>/dev/null && echo "killed vllm $VLLM_PID"
  [ -n "$INSERTER_PID" ] && kill "$INSERTER_PID" 2>/dev/null && echo "killed inserter $INSERTER_PID"
  sleep 3
  [ -n "$REDIS_PID" ]   && kill "$REDIS_PID"   2>/dev/null && echo "killed redis $REDIS_PID"
  [ -n "$MONGO_PID" ]   && kill "$MONGO_PID"   2>/dev/null && echo "killed mongo $MONGO_PID"
  sleep 2
  pkill -u "$USER" -f "redis-server" 2>/dev/null || true
  pkill -u "$USER" -f "mongod"       2>/dev/null || true
  date
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# Step 1: Start Redis
# --------------------------------------------------------------------------
echo
echo "== Step 1: start Redis (apptainer, host net, 127.0.0.1:6379) =="
apptainer run --cleanenv --env LC_ALL=C --env LANG=C \
  --bind /dev/shm --bind /work/nvme \
  "$REDIS_SIF" \
  redis-server --bind 127.0.0.1 --port 6379 --save "" --appendonly no \
  > "$LOGDIR/redis-${SLURM_JOB_ID}.log" 2>&1 &
REDIS_PID=$!
echo "redis launched pid=$REDIS_PID"
REDIS_OK=0
for i in $(seq 1 30); do
  if apptainer exec --cleanenv "$REDIS_SIF" \
       redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "redis PONG after ${i}s"
    REDIS_OK=1; break
  fi
  sleep 1
done
[ "$REDIS_OK" != "1" ] && { echo "FATAL: redis never PONG"; tail -20 "$LOGDIR/redis-${SLURM_JOB_ID}.log"; exit 1; }

# --------------------------------------------------------------------------
# Step 2: Start MongoDB
# --------------------------------------------------------------------------
echo
echo "== Step 2: start MongoDB (apptainer, node-local dbpath, 127.0.0.1:27017) =="
apptainer run --cleanenv --env LC_ALL=C --env LANG=C \
  --bind /dev/shm --bind /work/nvme \
  "$MONGO_SIF" \
  mongod --dbpath "$MONGO_DATA" --bind_ip 127.0.0.1 --port 27017 \
  --wiredTigerCacheSizeGB 1 \
  > "$LOGDIR/mongo-${SLURM_JOB_ID}.log" 2>&1 &
MONGO_PID=$!
echo "mongod launched pid=$MONGO_PID"
MONGO_OK=0
for i in $(seq 1 40); do
  if python3 -c "import socket; socket.create_connection(('127.0.0.1',27017),2)" 2>/dev/null; then
    echo "mongo port open after ${i}s"
    MONGO_OK=1; break
  fi
  sleep 1
done
[ "$MONGO_OK" != "1" ] && { echo "FATAL: mongo port 27017 never opened"; tail -30 "$LOGDIR/mongo-${SLURM_JOB_ID}.log"; exit 1; }
sleep 2  # extra grace for mongod to accept real commands

# --------------------------------------------------------------------------
# Step 3: Start Flowcept DocumentInserter (MQ -> MongoDB consumer)
#   Runs from the apptainer container (which has flowcept installed),
#   with PYTHONPATH pointing at the fixed kvnorm connector.
#   This is the SAME pattern as flowcept_pipeline.slurm (verified working).
# --------------------------------------------------------------------------
echo
echo "== Step 3: start Flowcept DocumentInserter (Redis MQ -> MongoDB) =="

BIND="--bind /work/nvme --bind /dev/shm --bind /u/jcernuda/spotterAI:/u/jcernuda/spotterAI"
ENVV_BASE="--env HF_HOME=$HF_HOME \
  --env HF_HUB_OFFLINE=0 \
  --env HF_HUB_DISABLE_XET=1 \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=$KVSRC:$PKG \
  --env VLLM_CACHE_ROOT=$SCRATCH/vllm_cache \
  --env TRITON_CACHE_DIR=$CH/triton \
  --env TORCHINDUCTOR_CACHE_DIR=$CH/inductor \
  --env XDG_CACHE_HOME=$CH/xdg \
  --env FLOWCEPT_SETTINGS_PATH=$SETTINGS \
  --env MQ_ENABLED=true \
  --env MQ_HOST=127.0.0.1 \
  --env MQ_PORT=6379 \
  --env MONGO_ENABLED=true \
  --env MONGO_HOST=127.0.0.1 \
  --env MONGO_PORT=27017"

RUN_CONTAINER="apptainer exec --cleanenv --nv --home $CH/home $BIND $ENVV_BASE $SIF"

# Start DocumentInserter in background: it blocks consuming Redis MQ until killed.
$RUN_CONTAINER python3 -c "
import time, sys
from flowcept import Flowcept
print('DocumentInserter: starting Flowcept persistence...', flush=True)
f = Flowcept(start_persistence=True)
f.start()
print('DocumentInserter: running (ctrl-c or SIGTERM to stop)...', flush=True)
try:
    while True:
        time.sleep(5)
except (KeyboardInterrupt, SystemExit):
    print('DocumentInserter: stopping...', flush=True)
    f.stop()
    print('DocumentInserter: stopped.', flush=True)
" > "$LOGDIR/inserter-${SLURM_JOB_ID}.log" 2>&1 &
INSERTER_PID=$!
echo "DocumentInserter launched pid=$INSERTER_PID"
sleep 4  # give inserter time to connect to Redis+Mongo before vLLM starts sending

# --------------------------------------------------------------------------
# Step 4: Verify connector loads + config is consistent
# --------------------------------------------------------------------------
echo
echo "== Step 4: verify fixed connector + Flowcept config =="
$RUN_CONTAINER python3 -c "
import vllm_kvnorm, inspect, pathlib
src = pathlib.Path(inspect.getfile(vllm_kvnorm))
print('vllm_kvnorm loaded from:', src)
import flowcept.configs as c
print('SETTINGS_PATH =', c.SETTINGS_PATH)
print('MQ_ENABLED', c.MQ_ENABLED, 'MONGO_ENABLED', c.MONGO_ENABLED, 'FLUSH', c.DB_FLUSH_MODE)
print('MQ', c.MQ_HOST, ':', c.MQ_PORT, '  MONGO', c.MONGO_HOST, ':', c.MONGO_PORT, 'db=', c.MONGO_DB)
"

# --------------------------------------------------------------------------
# Step 5: Start vllm serve with kvnorm connector
#   - VLLM_ENABLE_V1_MULTIPROCESSING=1 so the EngineCore subprocess inherits
#     FLOWCEPT_SETTINGS_PATH and can reach Redis MQ from inside the container
#   - --kv-transfer-config loads KVNormConnector
#   - workflow_id in extra_config provides the parent_workflow_id link
# --------------------------------------------------------------------------
echo
echo "== Step 5: start vllm serve (granite-4.2-30b + kvnorm connector) =="

KV_TRANSFER_CFG='{"kv_connector":"KVNormConnector","kv_connector_module_path":"vllm_kvnorm","kv_role":"kv_producer","kv_connector_extra_config":{"workflow_id":"spotter-fused-dataset"}}'

ENVV_VLLM="$ENVV_BASE \
  --env VLLM_ENABLE_V1_MULTIPROCESSING=1"

RUN_VLLM="apptainer exec --cleanenv --nv --home $CH/home $BIND $ENVV_VLLM $SIF"

$RUN_VLLM python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_HF_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --port "$VLLM_PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --kv-transfer-config "$KV_TRANSFER_CFG" \
  2>&1 | tee "$LOGDIR/vllm-${SLURM_JOB_ID}.log" &
VLLM_PID=$!
echo "vLLM serve PID: $VLLM_PID"
echo "KV-transfer-config: $KV_TRANSFER_CFG"

# --------------------------------------------------------------------------
# Step 6: Poll vLLM /health (model load can take ~10 min for 30B)
# --------------------------------------------------------------------------
echo
echo "== Step 6: poll vLLM /health (up to 12 min) =="
DEADLINE=$(( $(date +%s) + 720 ))
VLLM_READY=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
  echo "  vLLM health: HTTP $HTTP_CODE  ($(date +%H:%M:%S))"
  if [ "$HTTP_CODE" = "200" ]; then
    VLLM_READY=1
    echo "  vLLM READY."
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM server died early. Check $LOGDIR/vllm-${SLURM_JOB_ID}.log"
    tail -50 "$LOGDIR/vllm-${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 15
done
if [ "$VLLM_READY" -eq 0 ]; then
  echo "ERROR: vLLM did not become ready."
  tail -50 "$LOGDIR/vllm-${SLURM_JOB_ID}.log"
  exit 1
fi

echo
echo "== vLLM /v1/models =="
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" | python3 -m json.tool || true

# --------------------------------------------------------------------------
# Step 7: Run clio driver from headnode venv
#   CLIO_PROVENANCE_PROVIDERS=flowcept + FLOWCEPT_SETTINGS_PATH -> same Redis/Mongo
# --------------------------------------------------------------------------
echo
echo "== Step 7: run clio driver (multi-prompt, flowcept provenance enabled) =="

CLIO_USER_DIR="$CLIO_USER_DIR" \
CLIO_LM_PROVIDER="vllm" \
CLIO_LM_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1" \
CLIO_LM_MODEL="$SERVED_MODEL_NAME" \
CLIO_LM_API_KEY="EMPTY" \
CLIO_LM_MAX_TOKENS="1024" \
CLIO_LM_TEMPERATURE="0.0" \
CLIO_ARC_STORE="local" \
CLIO_LIVE_STREAMING="0" \
CLIO_PROVENANCE_PROVIDERS="flowcept" \
FLOWCEPT_SETTINGS_PATH="$SETTINGS" \
"$CLIO_VENV/bin/python" \
  "$SLURM_SUBMIT_DIR/clio_kvnorm_driver.py" \
  --gact-port "$GACT_PORT" \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
  --model "$SERVED_MODEL_NAME" \
  --workspace-root "$WS_ROOT" \
  2>&1 | tee "$LOGDIR/clio-${SLURM_JOB_ID}.log"

CLIO_EXIT=${PIPESTATUS[0]}
echo "clio driver exit: $CLIO_EXIT"

# Give vLLM a moment to flush any outstanding kvnorm records
echo "== waiting 15s for vLLM/kvnorm to drain remaining records =="
sleep 15

# --------------------------------------------------------------------------
# Step 8: Verify both streams in MongoDB
# --------------------------------------------------------------------------
echo
echo "== Step 8: verify BOTH provenance streams in MongoDB =="

$RUN_CONTAINER python3 - <<'PYEOF'
import flowcept.configs as c
from pymongo import MongoClient
import json

uri = f"mongodb://{c.MONGO_HOST}:{c.MONGO_PORT}"
print(f"Connecting to {uri}, db={c.MONGO_DB}")
client = MongoClient(uri, serverSelectionTimeoutMS=8000)
client.admin.command("ping")
db = client[c.MONGO_DB]

colls = db.list_collection_names()
print(f"\nCollections: {colls}")
for cn in colls:
    n = db[cn].count_documents({})
    if n > 0:
        print(f"  {cn}: {n} docs")

# --- kvnorm records ---
tasks = db["tasks"]
n_kv = tasks.count_documents({"activity_id": "kv_token_importance"})
print(f"\n=== kvnorm kv_token_importance tasks: {n_kv} ===")
kv_sample = tasks.find_one({"activity_id": "kv_token_importance"})
if kv_sample:
    used = kv_sample.get("used", {}) or {}
    gen = kv_sample.get("generated", {}) or {}
    meta = kv_sample.get("custom_metadata", {}) or {}
    score = gen.get("score", []) or []
    print(f"  SAMPLE kv_token_importance:")
    print(f"    workflow_id      = {kv_sample.get('workflow_id')!r}")
    print(f"    used.request_id  = {used.get('request_id')!r}")
    print(f"    used.num_computed_tokens = {used.get('num_computed_tokens')!r}")
    print(f"    len(score)       = {len(score)}")
    print(f"    metric           = {meta.get('metric')!r}")
    print(f"    num_layers       = {meta.get('num_layers')!r}")
    print(f"    status           = {kv_sample.get('status')!r}")

# --- clio records ---
n_clio_tasks = tasks.count_documents({"subtype": "clio_event"})
n_lm_tasks   = tasks.count_documents({"subtype": "ai_model_invocation"})
n_clio_all   = tasks.count_documents({"custom_metadata.clio": {"$exists": True}})
print(f"\n=== clio provenance tasks: ===")
print(f"  subtype=clio_event:           {n_clio_tasks}")
print(f"  subtype=ai_model_invocation:  {n_lm_tasks}")
print(f"  has custom_metadata.clio:     {n_clio_all}")

clio_sample = tasks.find_one({"custom_metadata.clio": {"$exists": True}})
if clio_sample:
    meta = clio_sample.get("custom_metadata", {}) or {}
    clio_m = meta.get("clio", {}) or {}
    print(f"  SAMPLE clio task:")
    print(f"    workflow_id      = {clio_sample.get('workflow_id')!r}")
    print(f"    activity_id      = {clio_sample.get('activity_id')!r}")
    print(f"    subtype          = {clio_sample.get('subtype')!r}")
    print(f"    clio.event_type  = {clio_m.get('event_type')!r}")
    print(f"    clio.session_id  = {clio_m.get('session_id')!r}")
    print(f"    clio.workspace_id= {clio_m.get('workspace_id')!r}")
    print(f"    status           = {clio_sample.get('status')!r}")

# --- workflows ---
workflows = db["workflows"]
n_wf = workflows.count_documents({})
n_agentic = workflows.count_documents({"subtype": "agentic_workflow"})
n_kvnorm_wf = workflows.count_documents({"parent_workflow_id": "spotter-fused-dataset"})
print(f"\n=== workflows: total={n_wf} agentic={n_agentic} kvnorm_child={n_kvnorm_wf} ===")

# --- join candidate analysis ---
print("\n=== JOIN CANDIDATE FIELDS ===")
print("  kvnorm records carry:  workflow_id (from kv_connector_extra_config), used.request_id")
print("  clio records carry:    workflow_id (from FlowceptProvenanceProvider._workflow_id(session_id))")
print("  Link: both write to same MongoDB (flowcept db). Temporal join on started_at.")
print("  kvnorm: workflow_id = vllm-kvnorm-<hex> (child of 'spotter-fused-dataset')")
print("  clio:   workflow_id = stable UUID5('clio:workflow-session:<session_id>')")
print("  Shared context: request timestamps in both sets align per-session.")

# --- final verdict ---
print("\n=== FINAL VERDICT ===")
both_present = (n_kv > 0) and (n_clio_all > 0 or n_lm_tasks > 0)
print(f"  kv_token_importance records: {n_kv}")
print(f"  clio provenance records:     {n_clio_all}")
print(f"  BOTH STREAMS IN MONGO:       {'YES' if both_present else 'NO'}")

client.close()
PYEOF

VERIFY_EXIT=$?
echo "mongo verify exit: $VERIFY_EXIT"

# --------------------------------------------------------------------------
# Step 9: Export dataset (mongodump + mongoexport + README)
# --------------------------------------------------------------------------
echo
echo "== Step 9: export dataset to $EXPORT_DIR =="

# Get clio repo commit for README
CLIO_COMMIT=$(cd /u/jcernuda/spotterAI/clio-agent && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
KVNORM_COMMIT=$(cd /u/jcernuda/spotterAI/vllm-kvnorm && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
CLIO_BRANCH=$(cd /u/jcernuda/spotterAI/clio-agent && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# mongodump (binary archive, gzip compressed)
DUMP_ARCHIVE="$EXPORT_DIR/spotter_fused_${SLURM_JOB_ID}.archive.gz"
apptainer exec --cleanenv --env LC_ALL=C --env LANG=C \
  --bind /work/nvme --bind /dev/shm \
  "$MONGO_SIF" \
  mongodump \
    --host 127.0.0.1 --port 27017 \
    --db flowcept \
    --archive="$DUMP_ARCHIVE" \
    --gzip
echo "mongodump: $DUMP_ARCHIVE"
ls -lh "$DUMP_ARCHIVE" 2>/dev/null

# mongoexport: tasks collection (kv_token_importance + clio records)
EXPORT_TASKS="$EXPORT_DIR/tasks_${SLURM_JOB_ID}.json"
apptainer exec --cleanenv --env LC_ALL=C --env LANG=C \
  --bind /work/nvme --bind /dev/shm \
  "$MONGO_SIF" \
  mongoexport \
    --host 127.0.0.1 --port 27017 \
    --db flowcept \
    --collection tasks \
    --out "$EXPORT_TASKS"
echo "mongoexport tasks: $EXPORT_TASKS"
ls -lh "$EXPORT_TASKS" 2>/dev/null

# mongoexport: workflows collection
EXPORT_WF="$EXPORT_DIR/workflows_${SLURM_JOB_ID}.json"
apptainer exec --cleanenv --env LC_ALL=C --env LANG=C \
  --bind /work/nvme --bind /dev/shm \
  "$MONGO_SIF" \
  mongoexport \
    --host 127.0.0.1 --port 27017 \
    --db flowcept \
    --collection workflows \
    --out "$EXPORT_WF"
echo "mongoexport workflows: $EXPORT_WF"
ls -lh "$EXPORT_WF" 2>/dev/null

# Collect counts for README
N_KV=$($RUN_CONTAINER python3 -c "
from pymongo import MongoClient
import flowcept.configs as c
db = MongoClient(f'mongodb://{c.MONGO_HOST}:{c.MONGO_PORT}', serverSelectionTimeoutMS=5000)[c.MONGO_DB]
print(db['tasks'].count_documents({'activity_id': 'kv_token_importance'}))
" 2>/dev/null || echo "?")

N_CLIO=$($RUN_CONTAINER python3 -c "
from pymongo import MongoClient
import flowcept.configs as c
db = MongoClient(f'mongodb://{c.MONGO_HOST}:{c.MONGO_PORT}', serverSelectionTimeoutMS=5000)[c.MONGO_DB]
print(db['tasks'].count_documents({'custom_metadata.clio': {'\$exists': True}}))
" 2>/dev/null || echo "?")

N_WF=$($RUN_CONTAINER python3 -c "
from pymongo import MongoClient
import flowcept.configs as c
db = MongoClient(f'mongodb://{c.MONGO_HOST}:{c.MONGO_PORT}', serverSelectionTimeoutMS=5000)[c.MONGO_DB]
print(db['workflows'].count_documents({}))
" 2>/dev/null || echo "?")

# Write README
cat > "$EXPORT_DIR/README.md" <<README_EOF
# SPOTTER-AI Fused Provenance Dataset

Generated by SLURM job ${SLURM_JOB_ID} on $(date -u +%Y-%m-%dT%H:%M:%SZ).

## Contents

### Record Counts (MongoDB 'flowcept' db)
- **kv_token_importance tasks** (kvnorm stream): ${N_KV}
- **clio provenance tasks** (clio agent stream): ${N_CLIO}
- **workflow documents** (both streams): ${N_WF}

### Files
- \`spotter_fused_${SLURM_JOB_ID}.archive.gz\` — mongodump binary archive (restore: see below)
- \`tasks_${SLURM_JOB_ID}.json\` — mongoexport NDJSON of the 'tasks' collection
- \`workflows_${SLURM_JOB_ID}.json\` — mongoexport NDJSON of the 'workflows' collection

## What's Inside

### Stream 1: kvnorm / vLLM KV-norm provenance
- Collection: \`tasks\`
- Filter: \`{"activity_id": "kv_token_importance"}\`
- Fields: \`workflow_id\`, \`used.request_id\`, \`used.num_computed_tokens\`,
  \`generated.score\` (per-token importance), \`custom_metadata.metric\`
- Emitted by: \`KVNormConnector\` (vllm-kvnorm) via Flowcept VLLMInterceptor

### Stream 2: clio-agent provenance
- Collection: \`tasks\`
- Filter: \`{"custom_metadata.clio": {"\$exists": true}}\`
- Subtypes: \`clio_event\`, \`ai_model_invocation\`
- Fields: \`workflow_id\`, \`activity_id\`, \`custom_metadata.clio.event_type\`,
  \`custom_metadata.clio.session_id\`, \`custom_metadata.clio.turn_id\`
- Emitted by: \`FlowceptProvenanceProvider\` in clio-agent via Flowcept

### Join / Cross-Stream Correlation
Both streams write to the same MongoDB database (flowcept) and share:
- **Temporal alignment**: both records carry \`started_at\` timestamps
- **Shared workflow root**: kvnorm emits under parent_workflow_id="spotter-fused-dataset";
  clio emits per-session workflow_ids
- **Request-level link**: \`used.request_id\` in kvnorm records matches vLLM's internal
  request_id (visible in vLLM server logs alongside clio's HTTP calls)

## Model & Code
- Model: ${MODEL_HF_ID} served as '${SERVED_MODEL_NAME}'
- vLLM: from ${SIF}
- clio-agent: branch=${CLIO_BRANCH}, commit=${CLIO_COMMIT}
- vllm-kvnorm connector: commit=${KVNORM_COMMIT}
- Flowcept settings: ${SETTINGS}

## Restore
\`\`\`bash
# Start MongoDB
mongod --dbpath /tmp/restore_db --bind_ip 127.0.0.1 --port 27017 &
# Restore
mongorestore --host 127.0.0.1 --port 27017 \\
  --archive=spotter_fused_${SLURM_JOB_ID}.archive.gz \\
  --gzip
# Query
mongosh --eval "db.getSiblingDB('flowcept').tasks.countDocuments({'activity_id':'kv_token_importance'})"
\`\`\`
README_EOF

echo "README written: $EXPORT_DIR/README.md"

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
echo
echo "== FINAL SUMMARY =="
echo "  kv_token_importance records: $N_KV"
echo "  clio provenance records:     $N_CLIO"
echo "  workflow docs:               $N_WF"
echo
echo "  KV-transfer-config used:"
echo "    $KV_TRANSFER_CFG"
echo
echo "  Export files:"
ls -lh "$EXPORT_DIR/" 2>/dev/null
echo
echo "  Logs:"
ls -lh "$LOGDIR/" 2>/dev/null
echo
echo "  clio driver exit: $CLIO_EXIT"
echo "  mongo verify exit: $VERIFY_EXIT"
echo
if [ "${N_KV}" != "0" ] && [ "${N_CLIO}" != "0" ] && [ "${N_KV}" != "?" ] && [ "${N_CLIO}" != "?" ]; then
  echo "  STATUS: BOTH STREAMS PRESENT IN MONGO -- dataset is complete."
else
  echo "  STATUS: WARNING -- one or more streams may be missing (N_KV=${N_KV} N_CLIO=${N_CLIO})"
fi
echo
date
exit 0
