# ImputeForge — Makefile
# Author: 晨星

PY ?= python
RUFF ?= $(PY) -m ruff
PKG := imputeforge

.PHONY: help install lint format check test cov demo detcheck clean lock

help: ## show this help
	@echo "ImputeForge targets"
	@echo "  make install   install runtime + dev dependencies"
	@echo "  make lint      ruff check"
	@echo "  make format    ruff format"
	@echo "  make check     lint + format --check + pytest  (local CI equivalent)"
	@echo "  make test      pytest -q"
	@echo "  make cov       pytest with coverage on $(PKG)"
	@echo "  make demo      run the end-to-end demo -> examples/benchmark.json"
	@echo "  make detcheck  determinism check (same seed twice)"
	@echo "  make bench     full benchmark via the CLI"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest ruff

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=$(PKG) --cov-report=term-missing

check: ## local CI equivalent — must be green before any push
	$(RUFF) check .
	$(RUFF) format --check .
	$(PY) -m pytest -q

demo:
	$(PY) examples/run_demo.py

detcheck:
	$(PY) cli.py detcheck

bench:
	$(PY) cli.py benchmark --full-grid

lock:
	$(PY) -m pip freeze | grep -viE "^(pip|setuptools|wheel)" > requirements.freeze.txt
	@echo "wrote requirements.freeze.txt — hand-check it before shipping"

clean:
	rm -rf .pytest_cache .ruff_cache .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
