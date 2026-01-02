install:
	pip install uv &&\
	uv sync --no-dev

test:
	uv run --no-dev python -m pytest tests/ -vv --cov=custom_lib --cov=api --cov=cli

format:	
	uv run --no-dev black custom_lib/*.py api/*.py 

lint:
	uv run --no-dev pylint --disable=R,C,E0401 --ignore-patterns=test_.*\.py custom_lib/*.py api/*.py

refactor: format lint

all: install refactor test