# Distill-SQL

**A 1.5B text-to-SQL model distilled from large LLMs. It runs on a laptop.**

> 🚧 Work in progress. Numbers below are placeholders until the evaluation phases are complete.
> The full design record is in [docs/PLAN.md](docs/PLAN.md).

## The problem

Large LLMs write good SQL, but every query costs money, sends your schema to a third party, and takes
seconds. Most analytics questions are routine. A small local model could answer them, and the
expensive model would only be needed for the hard cases.

Distill-SQL fine-tunes **Qwen2.5-Coder-1.5B-Instruct** on SQL that larger teacher models generated
and that we **checked by running it against real databases**. Then it measures how much of the teacher's
accuracy survives, and what that costs.

## Results (placeholder)

| Model | Spider dev EX | BIRD Mini-Dev EX | Latency p50 | Cost / 1k queries | Runs on laptop |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B (base) | – | – | – | ~$0 | ✅ |
| + SFT (ours) | – | – | – | ~$0 | ✅ |
| + SFT + DPO (ours) | – | – | – | ~$0 | ✅ |
| + SFT + DPO, GGUF Q4_K_M (ours) | – | – | – | ~$0 | ✅ |
| Teacher (gpt-oss-120b) | – | – | – | $– | ❌ |

EX = execution accuracy (the predicted SQL returns the same result as the gold SQL).

## Architecture

```mermaid
flowchart LR
  A[Spider train DBs] --> B[Schema extraction]
  B --> C[Teacher LLMs<br/>Groq / Gemini free tier]
  C --> D[Execute & verify<br/>on SQLite]
  D -->|correct| E[SFT data]
  D -->|wrong| F[DPO negatives]
  E --> G[LoRA SFT<br/>Kaggle T4]
  G --> H[DPO]
  F --> H
  H --> I[Merge → GGUF 4-bit]
  I --> J[FastAPI service]
  J --> K[Router: small model first,<br/>teacher fallback]
  I --> L[Gradio demo]
  X[(Contamination check)] -. CI gate .-> E
```

## Repository layout

```
configs/           YAML configs (validated by pydantic): data, generation, training, eval, serving
src/distill_sql/   the Python package
  config.py        typed config loading
  data/            download, schema extraction, contamination check
  generation/      teacher clients, rate limiting, caching
  verification/    execute + filter + dedupe generated SQL
  training/        dataset builders for SFT / DPO (GPU code lives in notebooks/)
  evaluation/      metrics, eval harness, reports
  serving/         FastAPI app
  router/          small-model-first router for sql-analyst-agent
notebooks/         Kaggle-ready training notebooks
demo/              Gradio app for Hugging Face Spaces
tests/             pytest suite (+ fixtures for the CI eval regression check)
docs/              plan, design notes
data/, runs/       local only, git-ignored
```

## Quickstart

```bash
# 1. install (uses uv: https://docs.astral.sh/uv/)
make install

# 2. secrets
cp .env.example .env   # then add your free Groq / Gemini keys

# 3. checks
make test

# 4. data (Phase 1): downloads ~1 GB, builds data/processed/ in about 5 s after download
make data
make check-contamination

# 5. synthetic data (Phase 2): resumable, re-run once a day until enough data
make generate-dry                 # example prompt + rate-limit budget, no API calls
make generate MAX_CALLS=5         # small pilot
make generate                     # all teachers in parallel, stops cleanly at daily limits
make verify                       # execute + filter + dedupe -> data/processed/train/sft.jsonl
make crosscheck CHECKER=gemini-flash   # optional: second teacher re-answers kept questions
make verify && make check-contamination
```

Outputs: `data/processed/registry.json` (every DB with its train/eval/excluded role),
`examples/` (all questions in one format), `schemas/` (per-DB schema + sample rows) and
`reports/data_report.md`. Phase 2 adds `train/sft.jsonl`, `rejected/rejected.jsonl` and
`reports/generation_report.md` (keep rates, rejection reasons, SQL complexity, cost at list price).
Run `make help` for every command.

## Data and licences

| Asset | Licence | Use here |
|---|---|---|
| [Spider 1.0](https://yale-lily.github.io/spider) | CC BY-SA 4.0 | Train-split DBs for generation; dev/test for evaluation only |
| [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) | CC BY-SA 4.0 | Evaluation only |
| [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) | Apache-2.0 | Spider hardness and exact-match scoring |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | Apache-2.0 | Base model |

Synthetic datasets derived from these databases are released under **CC BY-SA 4.0** (ShareAlike).
No dataset files are committed to this repository.

**No test-set contamination:** training data is generated only from Spider *train* databases.
`make check-contamination` fails if any eval database, schema or question appears in the training
data (it compares names, schema fingerprints, exact and near-duplicate questions, and 13-grams). CI
runs the checker's unit tests. On the real data it caught one cross-dataset leak: Spider's
*train* `formula_1` is the same database as BIRD Mini-Dev's `formula_1`. That database is excluded from training.

| Split | Role | Questions | DBs |
|---|---|---:|---:|
| Spider train (`train_spider` + `train_others`) | databases for synthetic generation | (8,659 not used as labels) | 138 usable (8 excluded) |
| Spider dev | eval: model selection | 1,034 | 20 |
| Spider test | eval: final, run once | 2,147 | 40 |
| BIRD Mini-Dev (SQLite) | eval: harder, out-of-domain | 500 | 11 |

## Design decisions

See [docs/PLAN.md](docs/PLAN.md) for the full list. Short version:
- **Execution-verified distillation.** Teacher SQL is kept only if it runs and returns a plausible result.
- **Synthetic-only training data.** No Spider train labels, so the gains come from the teachers.
- **Cost is a first-class metric.** Every result row has a cost per 1,000 queries.

## Limitations

To be completed in Phase 10. Known now: public benchmarks may already be in the pretraining data of
both the base model and the teachers, and we can only measure SQLite.

## Cost breakdown

| Item | Cost |
|---|---|
| Teacher API calls (Groq / Gemini free tiers) | $0 |
| Training (Kaggle free GPU) | $0 |
| Tracking (MLflow local) | $0 |
| Hosting (Hugging Face Hub + Spaces free tier) | $0 |
| **Total** | **$0** |

## Reproduce

To be written as phases land. Every run saves its resolved config, seed, git SHA and metrics to `runs/<run_id>/`.
