.PHONY: init venv format

PYTHON = python3.5
SHELL = bash
VENV_PATH = venv

init:
	find .git/hooks -type l -exec rm {} \;
	find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
	mkdir output/
	mkdir output/checkpoints/
	mkdir output/fonts/

venv:
	rm -Rf ${VENV_PATH}
	bash env_setup.sh ${PYTHON} ${VENV_PATH}

format:
	yapf --in-place -r src/ -e src/serialization/font_pb2.py -e src/serialization/font.proto
