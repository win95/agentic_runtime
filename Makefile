.PHONY: install install-requirements run-api run-worker run-beat migrate test lint docker-up docker-down seed seed-dry-run ingest ingest-dry-run demo eval eval-real eval-genai eval-genai-real langgraph-dev gen-synthetic-data eval-tool-use eval-retrieval train-embedding train-llm

# Prefer the project venv when it exists (this checkout), else system python.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

install:
	pip install -e ".[dev]"

# Pinned snapshot for reproducible local installs (see requirements.txt).
install-requirements:
	pip install -r requirements.txt

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

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

test:
	pytest -q

lint:
	ruff check app runtime infra ingest tests seeds scripts alembic

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
