.PHONY: env html clean

env:
	python -m pip install -e .
	python -m pip install -U mystmd pytest

html:
	myst build --html

clean:
	rm -rf _build
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
