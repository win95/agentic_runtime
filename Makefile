.PHONY: install install-requirements run-api run-worker run-beat migrate migrate-users run-user-service seed-admin test lint docker-up docker-down seed seed-dry-run seed-edge ingest ingest-dry-run demo eval eval-real eval-genai eval-genai-real langgraph-dev gen-synthetic-data eval-tool-use eval-retrieval train-embedding train-llm mlflow-report

# Prefer the project venv when it exists (this checkout), else system python.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

install:
	pip install -e ".[dev]"

# Pinned snapshot for reproducible local installs (see requirements.txt).
install-requirements:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# User service (independent service, same repo) — port 8001.
run-user-service:
	uvicorn services.user_service.main:app --host 0.0.0.0 --port 8001 --reload

run-worker:
	celery -A app.worker.celery_app worker --loglevel=info

# Celery beat — fires run_due_schedules every 60s. Schedules never fire
# without it: `make run-worker` alone only executes, it does not tick.
# Local runs need both processes (two terminals); in Docker the `beat`
# service in docker-compose.yml covers it.
run-beat:
	celery -A app.worker.celery_app beat --loglevel=info

# Apply Alembic migrations (alembic/versions). Fresh DB: upgrade head.
# A DB created by init_db()/create_all before migrations existed gets
# `alembic stamp head` once instead — see alembic/env.py.
migrate:
	$(PYTHON) -m alembic upgrade head

# User-service migrations (own chain, own DB — see
# services/user_service/README.md). Run from services/user_service.
migrate-users:
	cd services/user_service && $(PYTHON) -m alembic upgrade head

# One-off admin seed for the user service. Needs ADMIN_EMAIL and
# ADMIN_PASSWORD in the environment. Idempotent.
seed-admin:
	$(PYTHON) -m services.user_service.seed_admin

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check app runtime infra ingest ontology tests seeds scripts alembic services

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

# Seed demo data (asset O21, October 2026): missions + agents +
# schedules + Postgres rows + Qdrant docs + FalkorDB relations.
# Idempotent — re-running changes nothing. Needs the stores up
# (docker compose up -d) and the LiteLLM proxy for Qdrant embeddings.
seed:
	$(PYTHON) -m seeds.seed_demo_data

# Validate the seed data without touching any store.
seed-dry-run:
	$(PYTHON) -m seeds.seed_demo_data --dry-run

# Opt-in edge-case seeds (NOT part of `make seed`): plant one poisoned
# projection row (goes dead after 5 drains) for live drain/dead-path
# checks. Idempotent — re-running plants nothing.
seed-edge:
	$(PYTHON) -m seeds.seed_edge_cases

# Batch ingest (Option B): process JSON files in ingest/inbox.
# See docs/ingest.md. Retries pending quarantined rows:
#   $(PYTHON) -m ingest.run --retry
ingest:
	$(PYTHON) -m ingest.run --inbox ingest/inbox

# Validate ingest files without writing to any store.
ingest-dry-run:
	$(PYTHON) -m ingest.run --inbox ingest/inbox --dry-run

# End-to-end smoke test against a running API (needs the 4 LiteLLM vars).
demo:
	$(PYTHON) scripts/smoke_e2e.py

# MLflow eval, mock LLM (no key/proxy needed).
eval:
	$(PYTHON) scripts/eval_mission.py --mock-llm

# Adversarial eval (Layer 6 prompt-injection cases, mock mode, no MLflow).
eval-adversarial:
	$(PYTHON) scripts/eval_mission.py --mock-llm --adversarial

# MLflow eval with the real LiteLLM model (needs the 4 LiteLLM vars).
eval-real:
	$(PYTHON) scripts/eval_mission.py --real

# GenAI eval: classic eval + mlflow.genai.evaluate with a judge routed
# through the LiteLLM proxy (needs LLM_API_BASE/LLM_API_KEY; predictions
# still use the mock LLM unless --real is passed).
eval-genai:
	$(PYTHON) scripts/eval_mission.py --mock-llm --genai

# GenAI eval with real predictions (needs the 4 LiteLLM vars).
eval-genai-real:
	$(PYTHON) scripts/eval_mission.py --real --genai

# Rebuild the MLflow report for the manual hotel runs (read-only; needs
# the MLflow server only). Outputs under reports/ (git-ignored).
mlflow-report:
	$(PYTHON) scripts/manual/mlflow_report.py

# LangGraph dev server (Studio + traces) — http://localhost:8123
langgraph-dev:
	langgraph dev --allow-blocking

# --- Fine-tuning lifecycle (docs/fine-tuning-lifecycle.md) ---
# Synthetic data only — no real data yet. Train targets need torch/GPU
# (pip install -r training/requirements-training.txt on the training
# machine) and a real base model in training/configs/*.yaml; without
# them they stop with a clear message.

gen-synthetic-data:
	$(PYTHON) -m training.common.synthetic --out-dir training/data

eval-tool-use:
	$(PYTHON) -m training.llm.eval_tool_use --data training/data/synthetic_tool_use.jsonl --mock-predictions --no-mlflow

eval-retrieval:
	@test -n "$(MODEL)" || (echo "Usage: make eval-retrieval MODEL=<model> [BASELINE=<model>]" && exit 2)
	$(PYTHON) -m training.embedding.eval_retrieval --data training/data/synthetic_embedding.jsonl --model $(MODEL) $(if $(BASELINE),--baseline-model $(BASELINE)) --no-mlflow

train-embedding:
	$(PYTHON) -m training.embedding.train --config training/configs/embedding.yaml

train-llm:
	$(PYTHON) -m training.llm.train_sft --config training/configs/llm_sft.yaml
