# Temporal-join demonstration — clio agent messages ↔ kvnorm KV-importance

Fused dataset job **3082983** (Granite-4.2-30b on GH200). Both provenance streams
landed in one Flowcept MongoDB. They share **no direct key yet** — this alignment is
by **time-order** (both streams are 1:1, sequential, timestamped). Stage 3 will stamp
the vLLM response id onto the clio LM record for a bulletproof key.

**10 LM calls ↔ 10 kvnorm records**, across 5 agent sessions (prompts).

| # | prompt (agent turn) | turn | clio LM logged_at | kvnorm request_id | KV tokens | score len | score mean (min→max) |
|--|--|--|--|--|--|--|--|
| 1 | What is 17 multiplied by 23? Show your reasoni | completed | 2026-09-04T13:52:27.713Z | `chatcmpl-80fdc07ecb38f335-80b6501f:g0` | 7835 | 7835 | 9.25928 (4.42854→38.932) |
| 2 | Explain the difference between a stack and a q | completed | 2026-09-04T13:52:50.5Z | `chatcmpl-b87a8d1e2166bde7-a58480df:g0` | 7868 | 7868 | 9.30549 (4.42981→38.80653) |
| 3 | Explain the difference between a stack and a q | completed | 2026-09-04T13:53:10.796Z | `chatcmpl-a8a7b352bb323ebc-a959057a:g0` | 7905 | 7905 | 9.37644 (4.42349→38.37525) |
| 4 | Explain the difference between a stack and a q | completed | 2026-09-04T13:53:29.922Z | `chatcmpl-b525776416228c7f-82fe793b:g0` | 2310 | 2310 | 9.06712 (4.53898→26.28071) |
| 5 | Explain the difference between a stack and a q | completed | 2026-09-04T13:53:40.12Z | `chatcmpl-84d9290ada4cad4c-985e2de2:g0` | 2324 | 2324 | 9.00323 (4.51821→28.30675) |
| 6 | What are the three laws of thermodynamics? Giv | completed | 2026-09-04T13:54:02.604Z | `chatcmpl-943a2a52632892c1-9acb968e:g0` | 7870 | 7870 | 9.30285 (4.42709→38.8989) |
| 7 | What are the three laws of thermodynamics? Giv | completed | 2026-09-04T13:54:22.439Z | `chatcmpl-b31cfad78b336cf6-a51d9b8d:g0` | 8514 | 8514 | 9.25191 (4.42709→38.8989) |
| 8 | If a train travels at 60 mph for 2.5 hours, ho | failed | 2026-09-04T13:54:44.696Z | `chatcmpl-a2b17a248e1b4ce6-9e51df64:g0` | 7877 | 7877 | 9.35979 (4.42925→38.60389) |
| 9 | Write a haiku about machine learning. | failed | 2026-09-04T13:55:05.771Z | `chatcmpl-bc9cca6a4064fcd5-893ff3ea:g0` | 7860 | 7860 | 9.19925 (4.17409→38.68227) |
| 10 | Write a haiku about machine learning. | failed | 2026-09-04T13:55:24.995Z | `chatcmpl-a428e86baacd8cef-8645eb4b:g0` | 2269 | 2269 | 9.12003 (4.56071→31.47848) |

## How to read it
- Each row is one LM call the agent made: the **prompt/session** it served (clio side)
  aligned to the **per-token KV-importance array** vLLM+kvnorm produced for it.
- `KV tokens` = `num_computed_tokens` (prompt+generated); `score len` matches it —
  one importance float per token (mean over 64 Granite layers & KV heads,
  metric = PagedEviction ‖V‖₂/‖K‖₂).
- Multi-row sessions (e.g. prompt 2) are **multi-step ReAct loops** — the KV cache
  grows across steps, so later calls score longer token arrays.

## Caveat (why Stage 3)
Time-order pairing is correct here only because requests ran strictly sequentially
(vLLM `Running: 1 req`). Under concurrency it would be ambiguous. Recording the vLLM
`chatcmpl-*` response id on clio's `ai_model_invocation` gives a direct, robust join.
