# Actae benchmarks

The measurement harnesses and committed reports behind two load-bearing claims
about [Actae](https://actae.dev), a durable execution substrate for AI agents:

1. **Fork fidelity** — forking a run at an intermediate step (inheriting the
   prefix) produces output as faithful to the canonical run as re-running from
   scratch, while skipping the prefix.
2. **Crash consistency** — a side effect coordinated through Actae's execution
   ledger does not fire twice across a real process crash, down to the single
   ambiguous window (effect completed, response lost) that only a status query
   can close.

This repository is a **mirror** of the harnesses and reports that live in the
Actae source repository; it exists so the evidence is inspectable and
re-runnable without access to the private server code. The harnesses are MIT.
The Actae server is proprietary; the **Python/Go/TypeScript SDKs are MIT**.

## Fork fidelity

Five unrelated agent scenarios are each run twice — once forking at an
intermediate step, once from scratch — and a judge model at temperature 0
scores each final output against the canonical run. Higher is better. Across
every SDK and adapter the fork path is at least as faithful as the re-run:

| Adapter | Fork mean | Re-run mean |
|---|---|---|
| Python (session) | 8.47 | 7.07 |
| Go | 8.20 | 7.67 |
| TypeScript (session) | 8.80 | 8.20 |
| LangGraph (Python) | 8.40 | 7.80 |
| LangGraph (TypeScript) | 8.40 | 7.80 |

Raw reports: `reports/fork-fidelity-*.json` (each contains every judge
rationale and sample).

Reproduce:

```bash
pip install -r requirements.txt
# start a free self-hosted Actae: https://actae.dev/download
export ACTAE_URL=http://127.0.0.1:8002 ACTAE_API_KEY=sk-...
export DEEPSEEK_API_KEY=...            # OpenAI-compatible pipeline + judge
python harness/fork_fidelity_suite.py --repeats 3 --judge-reps 1 --report-dir reports/local
```

## Crash consistency

A worker is killed with `os._exit(137)` mid-run against a real local HTTP
service that exposes an idempotency key and a status query. The matrix below is
the worst case — the `non_idempotent` downstream — for a 6-step run crashing at
step 3. **Duplicates** is how many effects were applied more than once (0 is
the goal).

| Strategy | after_effect | after_complete | transport_drop | transport_fail |
|---|---|---|---|---|
| naive | 4 | 4 | 4 | 3 |
| checkpoint | 1 | 0 | 1 | 0 |
| durable | 1 | 0 | 1 | 0 |
| durable_query | 0 | 0 | 0 | 0 |

Across the every-step sweep (24 runs per strategy, all crash points, both
services): `durable` produced 18 duplicate applications; `durable_query`
produced 0. Raw reports: `reports/durability-results.json`,
`reports/durability-sweep.json`.

Reproduce:

```bash
pip install -r requirements.txt
export ACTAE_URL=http://127.0.0.1:8002 ACTAE_API_KEY=sk-...
python harness/durability_benchmark.py --services sqlite http \
  --out reports/local/durability-results.json
```

## How to read this honestly

- Fork fidelity is a **small, LLM-judged sample** (five scenarios, one judge at
  temperature 0). It is directional evidence that the fork path is not worse
  than re-running — not a statistically powerful result.
- Crash consistency is measured against a downstream that either honours the
  idempotency key or exposes a status query. Real systems also contain
  downstreams that do neither, and this benchmark does not claim otherwise.
  **Exactly-once external effects always require a cooperative downstream;**
  Actae shrinks the window, it does not erase the requirement.
- The harnesses exercise the open-source Python SDK. The server that stores the
  events, state, and ledger is separate and proprietary.

## Provenance

Mirrored from the Actae source repository by `scripts/publish-benchmarks-repo.sh`.
Report schemas and methodology are documented at
[actae.dev/docs/benchmarks](https://actae.dev/docs/benchmarks/) and
[actae.dev/docs/reliability](https://actae.dev/docs/reliability/).
