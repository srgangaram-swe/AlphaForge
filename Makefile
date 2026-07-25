PYTHON ?= python
UV ?= uv

.PHONY: install install-all lock-check test lint format typecheck policy check download-data build-features train evaluate \
        walk-forward backtest signal-foundry paper dashboard api report demo docker-build clean \
        native bench bench-native

install:
	$(UV) sync --locked --extra dev

install-all:
	$(UV) sync --locked --all-extras

lock-check:
	$(UV) lock --check

test:
	$(PYTHON) -m pytest -m "not network" --cov=alphaforge --cov-branch --cov-report=term-missing --cov-fail-under=78

lint:
	$(PYTHON) -m ruff check alphaforge tests scripts apps
	$(PYTHON) -m black --check alphaforge tests scripts apps

format:
	$(PYTHON) -m ruff check --fix alphaforge tests scripts apps
	$(PYTHON) -m black alphaforge tests scripts apps

typecheck:
	$(PYTHON) -m mypy alphaforge tests scripts apps

policy:
	$(PYTHON) -m pre_commit run --all-files

check: lock-check policy typecheck test

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

# Usage: make signal-foundry BUNDLE=/absolute/path/to/<bundle-id>
signal-foundry:
	@test -n "$(BUNDLE)" || (echo "BUNDLE must name a verified Signal Foundry bundle" >&2; exit 2)
	$(PYTHON) scripts/run_signal_foundry_research.py "$(BUNDLE)"

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

# --- C++ execution core ---
native:
	$(PYTHON) scripts/build_native.py

bench:
	$(PYTHON) scripts/bench_orderbook.py

bench-native:
	cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --target bench_orderbook
	./build/bench_orderbook

docker-build:
	docker build -t alphaforge:latest .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
