.PHONY: install run-api run-worker test lint docker-up docker-down seed

install:
	pip install -e ".[dev]"

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-worker:
	celery -A app.worker.celery_app worker --loglevel=info

test:
	pytest -q

lint:
	ruff check app runtime infra tests seeds

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

seed:
	python -m seeds.load_missions
