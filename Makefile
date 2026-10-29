.PHONY: all test results paper clean
all: test results paper

test:
	python -m pytest -q

results:
	python scripts/audit_original.py
	python scripts/run_all.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd paper && latexmk -C
