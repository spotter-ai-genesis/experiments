# experiments
Experiemental scratch space to assemble components needed in SPOTTER-AI

## OPAL attention-provenance experiment

Measuring where an LLM's attention goes when its evidence has been poisoned,
on the OPAL multi-round plant-phenotyping benchmark.

- [OPAL.md](OPAL.md) — what it is and how it is built
- [RUNBOOK.md](RUNBOOK.md) — how to run it on a GPU box, step by step
- [METHODS.md](METHODS.md) — each poisoning strategy, and what it measured
- [RESULTS.md](RESULTS.md) — findings from the 2026-10-27 suite

Benchmark data and the upstream reduction live in `bench/opal/`; the harness
is `scripts/`, `src/opal_attn/` and `configs/`.
