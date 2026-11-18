.ONESHELL:

all: format run

format:
	@ruff format src/sparse_bregman

run:
	@python3 src/main.py

run:
	@python3 src/eval.py

profile:
	@./cuda_profiler.sh