.PHONY: cpuvenv gpuvenv format 

PYTHON = python3.5
SHELL = bash

cpuvenv:
	( \
	rm -Rf cpuvenv; \
	${PYTHON} -m venv cpuvenv; \
	source cpuvenv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements/cpu.txt; \
	deactivate; \
	)

gpuenv:
	( \
	rm -Rf gpuvenv; \
	${PYTHON} -m venv gpuvenv; \
	source gpuvenv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements/gpu.txt; \
	deactivate; \
	)

format:
	yapf --in-place src/*.py
