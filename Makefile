# ─────────────────────────────────────────────────────────────────────────────
# PFAS-ARIA Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev setup lint format test test-cov \
        local-gate github-gate frontend-check security-check \
        run-pipeline run-api run-frontend dvc-init clean \
        db-up db-down db-logs db-reset

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
RUFF ?= $(VENV)/bin/ruff
PYTEST ?= $(VENV)/bin/pytest
MYPY ?= $(VENV)/bin/mypy
MYPY_FLAGS ?= --ignore-missing-imports --follow-imports=skip --cache-dir=/dev/null --disable-error-code=import-untyped
BANDIT ?= $(VENV)/bin/bandit
PIP_AUDIT ?= $(VENV)/bin/pip-audit
SEMGREP ?= $(VENV)/bin/semgrep
PRE_COMMIT ?= $(VENV)/bin/pre-commit
FRONTEND_DIR ?= frontend
DOCKER_IMAGE ?= pfas-aria:local

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
	@echo "  make local-gate    Fast local quality gate for debugging"
	@echo "  make github-gate   Full local mirror of GitHub Actions before push"
	@echo "  make security-check Run local Python security checks"
	@echo "  make frontend-check Run frontend lint/build/audit checks"
	@echo "  make run-pipeline  Run the full ARIA pipeline"
	@echo "  make run-api       Start the FastAPI backend (dev mode)"
	@echo "  make run-frontend  Start the React frontend (dev mode)"
	@echo "  make dvc-init      Initialise DVC for data versioning"
	@echo "  make db-up         Start local database stack (Postgres, Mongo, Redis)"
	@echo "  make db-down       Stop local database stack"
	@echo "  make db-logs       Follow database logs"
	@echo "  make db-reset      Wipe all local database volumes"
	@echo "  make clean         Remove all generated output files"
	@echo ""

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pre-commit
	$(PRE_COMMIT) install
	@echo "✅ Dev environment ready"

setup: install dvc-init
	$(PYTHON) -c "from src.utils.paths import ensure_dirs; ensure_dirs()"
	cp -n .env.example .env || true
	@echo "✅ Project setup complete — fill in your .env file"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html

security-check:
	$(BANDIT) -r src/ -ll
	$(PIP_AUDIT) -r requirements.txt --progress-spinner off

frontend-check:
	cd $(FRONTEND_DIR) && npm ci
	cd $(FRONTEND_DIR) && npm run lint
	cd $(FRONTEND_DIR) && npm run build
	cd $(FRONTEND_DIR) && npm audit --audit-level=critical

local-gate:
	$(PIP) check
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/
	$(MYPY) src/ $(MYPY_FLAGS)
	$(PYTHON) -c "from src.api.main import app; print('FastAPI app loaded OK')"
	$(PYTEST) tests/ -v --tb=short
	$(BANDIT) -r src/ -ll
	$(PIP_AUDIT) -r requirements.txt --progress-spinner off

github-gate: install
	$(PIP) check
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/
	$(MYPY) src/ $(MYPY_FLAGS)
	$(BANDIT) -r src/ -ll --exit-zero
	$(PIP_AUDIT) -r requirements.txt --progress-spinner off
	-$(SEMGREP) --config p/python --config p/secrets --config p/owasp-top-ten src/
	$(PYTHON) -c "from src.api.main import app; print('FastAPI app loaded OK')"
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=xml
	cd $(FRONTEND_DIR) && npm ci
	cd $(FRONTEND_DIR) && npm run lint
	cd $(FRONTEND_DIR) && npm run build
	cd $(FRONTEND_DIR) && npm audit --audit-level=critical
	docker build -t $(DOCKER_IMAGE) .

run-pipeline:
	$(PYTHON) -m src.orchestration.pipeline

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

db-down:
	docker compose down
	@echo "✅ Database stack stopped"

db-logs:
	docker compose logs -f

db-reset:
	docker compose down -v
	@echo "✅ All database volumes wiped"

dvc-init:
	@command -v dvc >/dev/null 2>&1 || $(PIP) install dvc
	dvc init --no-scm 2>/dev/null || dvc init
	dvc add data/raw data/corpus 2>/dev/null || true
	@echo "✅ DVC initialised"

clean:
	rm -rf data/outputs/logs/*
	rm -rf data/outputs/reports/*
	rm -rf mlflow/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"
