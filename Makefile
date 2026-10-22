.PHONY: all simulate analyze figures test clean
PYTHON ?= python3
REPS ?= 50
SEED ?= 1234

all: simulate analyze figures

simulate:
	$(PYTHON) simulate.py --reps $(REPS) --seed $(SEED)

analyze:
	$(PYTHON) analyze.py --stage analyze

figures:
	$(PYTHON) analyze.py --stage figures

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf results/raw/*.csv results/summary.csv results/figures/*.png results/figures/*.pdf


.PHONY: reproduce

reproduce:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	$(MAKE) PYTHON=.venv/bin/python test
	$(MAKE) PYTHON=.venv/bin/python all