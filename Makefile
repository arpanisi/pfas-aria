# ─────────────────────────────────────────────────────────────────────────────
# PFAS-ARIA Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev setup lint format test test-cov \
        run-pipeline run-api run-frontend dvc-init clean \
        db-up db-down db-logs db-reset

# Default: show help
help:
	@echo ""
	@echo "PFAS-ARIA — Available commands"
	@echo "────────────────────────────────────────"
	@echo "  make install       Install all Python dependencies"
	@echo "  make install-dev   Install + dev tools + pre-commit hooks"
	@echo "  make setup         Full first-time setup (install + dvc + dirs)"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Auto-format with ruff"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make run-pipeline  Run the full ARIA pipeline"
	@echo "  make run-api       Start the FastAPI backend (dev mode)"
	@echo "  make run-frontend  Start the React frontend (dev mode)"
	@echo "  make dvc-init      Initialise DVC for data versioning"
	@echo "  make db-up         Start local database stack (Postgres, Mongo, Redis, Chroma)"
	@echo "  make db-down       Stop local database stack"
	@echo "  make db-logs       Follow database logs"
	@echo "  make db-reset      Wipe all local database volumes"
	@echo "  make clean         Remove all generated output files"
	@echo ""

install:
	pip install -r requirements.txt

install-dev: install
	pip install pre-commit
	pre-commit install
	@echo "✅ Dev environment ready"

setup: install dvc-init
	python -c "from src.utils.paths import ensure_dirs; ensure_dirs()"
	cp -n .env.example .env || true
	@echo "✅ Project setup complete — fill in your .env file"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

run-pipeline:
	python -m src.orchestration.pipeline

run-worker:
	arq src.queue.worker.WorkerSettings

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm install && npm run dev

db-up:
	docker compose up -d
	@echo "✅ Database stack started"
	@echo "   PostgreSQL : localhost:5432"
	@echo "   MongoDB    : localhost:27017"
	@echo "   Redis      : localhost:6379"
	@echo "   ChromaDB   : localhost:8001"

db-down:
	docker compose down
	@echo "✅ Database stack stopped"

db-logs:
	docker compose logs -f

db-reset:
	docker compose down -v
	@echo "✅ All database volumes wiped"

dvc-init:
	@command -v dvc >/dev/null 2>&1 || pip install dvc
	dvc init --no-scm 2>/dev/null || dvc init
	dvc add data/raw data/corpus 2>/dev/null || true
	@echo "✅ DVC initialised"

clean:
	rm -rf data/outputs/logs/*
	rm -rf data/outputs/reports/*
	rm -rf data/outputs/chroma/*
	rm -rf mlflow/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"
