.PHONY: install dev test demo clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

demo:
	python -m kairos_clean audit ../kairos-data/data/ashare/sh600519.csv

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
