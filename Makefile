.PHONY: test lint typecheck check install clean

test:
	pytest -v

lint:
	ruff check src/ tests/

typecheck:
	mypy src/drumsep/

check: lint typecheck test

install:
	pip install -e ".[dev]"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .coverage htmlcov
