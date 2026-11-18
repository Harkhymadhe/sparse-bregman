.ONESHELL:

all: format run

format:
	@ruff format src/sparse_bregman

run:
	@python3 src/main.py