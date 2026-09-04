# clio × vLLM+kvnorm — fused provenance traces

Provenance traces from running the **clio-agent** on top of a **vLLM** server that
loads the **kvnorm** connector, with **both** provenance streams captured into a
single Flowcept MongoDB:

1. **clio agent provenance** — sessions, turns, ReAct steps, `ai_model_invocation`
   (LM calls), emitted by clio's `FlowceptProvenanceProvider`.
2. **kvnorm KV-importance** — per-token KV importance (`kv_token_importance`,
   PagedEviction ‖V‖₂/‖K‖₂ over all layers & KV heads), emitted by
   `KVNormConnector` via Flowcept's vLLM interceptor.

Both producers write only through the Flowcept library into one Redis MQ, drained
by one DocumentInserter into one Mongo — so the two streams are fused at capture.

## Layout

- [`simple/`](simple/) — the shareable dataset from a simple 5-prompt run
  (Granite-4.2-30b on a GH200): the `mongodump` archive, NDJSON exports, a dataset
  README, and a **temporal-join demonstration** ([`simple/temporal_join.md`](simple/temporal_join.md))
  pairing each clio agent LM call to the kvnorm token-importance array it produced.
- [`infrastructure/`](infrastructure/) — everything needed to reproduce it: the
  SLURM job, the clio driver, the Flowcept settings, and the container recipe.

## Status

- ✅ Both streams land in one Mongo (10 `kv_token_importance` ↔ 10 `ai_model_invocation`,
  1:1, from the same run).
- ⚠️ The streams share **no direct join key yet** — correlation today is by
  **time-order** (see `simple/temporal_join.md`). **Stage 3** will stamp the vLLM
  response id (`chatcmpl-*`) onto clio's LM record for a bulletproof join.
