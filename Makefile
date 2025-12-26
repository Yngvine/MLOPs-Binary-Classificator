install:
	pip install uv &&\
	uv sync --no-dev

test:
	uv run --no-dev python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli

format:	
	uv run --no-dev black model_lib/*.py api/*.py 

lint:
	uv run --no-dev pylint --disable=R,C,E0401 --generated-members=cv2.* --ignore-patterns=test_.*\.py model_lib/*.py api/*.py

refactor: format lint

all: install refactor test