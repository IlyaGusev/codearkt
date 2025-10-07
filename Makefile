.PHONY: install black validate test publish

install:
	uv pip install -e .[dev]

black:
	uv run black examples container codearkt tests --line-length 100

validate:
	uv run black examples container codearkt tests --line-length 100
	uv run flake8 examples container codearkt tests
	uv run mypy examples container codearkt tests --strict --explicit-package-bases

test:
	uv run pytest -s --cov=codearkt --cov-branch --cov-report=xml

publish:
	uv build && uv publish