# Infrastructure — how the fused dataset was produced

One SLURM job co-locates the whole stack on a single GH200 node (the cluster QOS
allows one running job/user, so serve + drive share the allocation):

```
                 ┌── vLLM serve (+ kvnorm connector) ──producer──┐
  one Redis MQ ←─┤                                                ├→ one DocumentInserter → one MongoDB
                 └── clio-agent (flowcept provider) ──producer────┘
```

## Files

| File | Role |
|---|---|
| `spotter_fused_dataset.slurm` | The job: starts Redis + Mongo + Flowcept DocumentInserter, `vllm serve` (with kvnorm), then the clio driver; exports the dataset; tears down. |
| `clio_kvnorm_driver.py` | Boots clio's GACT server in-process, binds the vLLM provider, sends the 5 prompts (one session each), waits for completion. |
| `flowcept_pipeline_settings.yaml` | Flowcept config (Redis MQ + Mongo, `db_flush_mode: online`, `kv_db.enabled: true`). **The same file is used by BOTH producers** — that shared pointer is what fuses the streams. Loopback, no auth. |
| `Dockerfile` / `vllm-kvnorm.def` | Container recipe: `vllm/vllm-openai:v0.28.0` + the Flowcept fork + `vllm_kvnorm`. Built to a `.sif` for Apptainer. |

## The kvnorm connector, in serve mode

vLLM loads kvnorm as a V1 KV connector via `--kv-transfer-config`:

```json
{"kv_connector":"KVNormConnector",
 "kv_connector_module_path":"vllm_kvnorm",
 "kv_role":"kv_producer",
 "kv_connector_extra_config":{"workflow_id":"spotter-fused-dataset"}}
```

Two things that are required and non-obvious:
- **`VLLM_ENABLE_V1_MULTIPROCESSING=1`** — so the EngineCore subprocess inherits
  `FLOWCEPT_SETTINGS_PATH` and can reach the same Redis MQ (otherwise kvnorm emits
  into a process that can't see the shared MQ).
- **`PYTHONPATH=<repo>/vllm-kvnorm/src`** — resolves `kv_connector_module_path`
  to the (fixed, hybrid-group-aware) connector.

## clio side (provider + provenance)

Environment the driver sets before booting GACT:

```
CLIO_LM_PROVIDER=vllm
CLIO_LM_API_BASE=http://127.0.0.1:8000/v1
CLIO_LM_MODEL=granite-4.2-30b        # == vLLM --served-model-name
CLIO_LM_API_KEY=EMPTY
CLIO_PROVENANCE_PROVIDERS=flowcept
FLOWCEPT_SETTINGS_PATH=<...>/flowcept_pipeline_settings.yaml   # SAME file as kvnorm
CLIO_ARC_STORE=local                 # scratch was tight; keep the ARC store local
CLIO_LIVE_STREAMING=0                 # Granite doesn't emit DSPy's [[## answer ##]] markers
```

clio branch: `codex/provider-runtime-campaign` (iowarp/clio-agent), venv built with
`uv sync --extra flowcept`.

## Restore the dataset

```bash
mongorestore --host 127.0.0.1 --port 27017 \
  --archive=../simple/spotter_fused_3082983.archive.gz --gzip
```

Or work directly from the NDJSON exports in `../simple/`.

## Regenerate the temporal join

```bash
python3 ../simple/make_temporal_join.py   # reads the NDJSON, writes temporal_join.{md,json}
```

## Known gaps (tracked for Stage 3)

- clio's `ai_model_invocation` record does **not** carry the vLLM `chatcmpl-*`
  response id, so cross-stream correlation is time-order only (see the caveat in
  `../simple/temporal_join.md`). Stage 3 records that id for a direct key.
- Some agent turns error on DSPy's `[[## answer ##]]` contract with a plain
  instruct model; the LM call still fires (so provenance + kvnorm are complete).
