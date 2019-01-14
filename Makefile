
.PHONY: cpuvenv gpuvenv format 

PYTHON = python3.5

cpuvenv:
	rm -Rf cpuvenv
	${PYTHON} -m venv cpuvenv
	$(shell source cpuvenv/bin/activate)
	pip install -r requirements/cpu.txt
	$(shell deactivate)

gpuenv:
	rm -Rf gpuvenv
	${PYTHON} -m venv gpuvenv
	$(shell source gpuvenv/bin/activate)
	pip install -r requirements/gpu.txt
	$(shell deactivate)

format:
	yapf --in-place src/*.py
