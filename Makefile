PYTHON ?= python
PIP ?= pip

.PHONY: install install-all test lint format typecheck download-data build-features train evaluate \
        walk-forward backtest paper dashboard api report demo docker-build clean

install:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[all]"

test:
	pytest -m "not network"

lint:
	ruff check alphaforge tests scripts apps
	black --check alphaforge tests scripts apps

format:
	ruff check --fix alphaforge tests scripts apps
	black alphaforge tests scripts apps

typecheck:
	mypy alphaforge

download-data:
	$(PYTHON) scripts/download_data.py --config configs/data.yaml

build-features:
	$(PYTHON) scripts/build_features.py --config configs/features.yaml

train:
	$(PYTHON) scripts/train_model.py --config configs/models.yaml

evaluate:
	$(PYTHON) scripts/evaluate_model.py --config configs/models.yaml

walk-forward:
	$(PYTHON) scripts/run_walk_forward.py --config configs/models.yaml

backtest:
	$(PYTHON) scripts/run_backtest.py --config configs/backtest.yaml

paper:
	$(PYTHON) scripts/simulate_paper_trading.py --config configs/backtest.yaml

dashboard:
	streamlit run apps/dashboard.py

api:
	uvicorn apps.api:app --reload --port 8000

report:
	$(PYTHON) scripts/generate_report.py

# Full end-to-end pipeline on synthetic data (no network required)
demo:
	$(PYTHON) scripts/run_walk_forward.py --synthetic --fast
	$(PYTHON) scripts/run_backtest.py --latest
	$(PYTHON) scripts/generate_report.py

docker-build:
	docker build -t alphaforge:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
