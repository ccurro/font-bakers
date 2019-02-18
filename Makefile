.PHONY: cpuvenv gpuvenv format 

PYTHON = python3.5
SHELL = bash

venv:
	( \
	rm -Rf venv; \
	${PYTHON} -m venv venv; \
	source venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	deactivate; \
	)

format:
	yapf --in-place -r src/
	yapf --in-place -r serialization/ -e serialization/font.proto -e serialization/font_pb2.py
