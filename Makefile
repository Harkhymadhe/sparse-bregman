.ONESHELL:

all: format run

format:
	@isort src/
	@ruff format src/

run:
	@python3 src/main.py

eval:
	@python3 src/eval.py

profile:
	@./cuda_profiler.sh