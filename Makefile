.PHONY: init venv format

PYTHON = python3.5
SHELL = bash

init:
	find .git/hooks -type l -exec rm {} \;
	find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
	mkdir output/
	mkdir output/checkpoints/
	mkdir output/fonts/

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
	yapf --in-place -r src/ -e src/serialization/font_pb2.py -e src/serialization/font.proto
