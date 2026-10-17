PY ?= .venv/bin/python
VENV_BIN := $(dir $(PY))

.PHONY: publish-check help install data pipeline quick synthetic results notebooks test cov lint format api ui monitor docker compose all clean
help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## venv + dev dependencies
	python3 -m venv .venv && $(VENV_BIN)pip install -q -r requirements-dev.txt

data:            ## download the two Home Credit CSVs (needs `kaggle auth login` and accepted competition rules)
	@for f in application_train.csv previous_application.csv; do \
	  $(VENV_BIN)kaggle competitions download -c home-credit-default-risk -f $$f -p data || exit 1; \
	  if [ -f data/$$f.zip ]; then $(PY) -m zipfile -e data/$$f.zip data && rm data/$$f.zip; fi; \
	  test -f data/$$f || { echo "data/$$f missing after download"; exit 1; }; \
	done

pipeline:        ## full run on the real data -> reports/ + models/ (resume-safe numbers)
	$(PY) -m src.pipeline --data-dir data

quick synthetic: ## plumbing check on synthetic data (NOT resume-safe)
	$(PY) -m src.pipeline --synthetic --quick

results:         ## write the README results block + print resume numbers from metrics.json
	$(PY) -m src.results_table --write-readme

notebooks:       ## execute the four notebooks in place (run after `make pipeline`)
	$(PY) notebooks/build_notebooks.py --execute

test:            ## unit, property and pipeline tests
	$(PY) -m pytest -q

cov:             ## tests with the coverage gate used in CI
	$(PY) -m pytest -q --cov --cov-report=term-missing

lint:            ## ruff lint + format check
	$(VENV_BIN)ruff check . && $(VENV_BIN)ruff format --check .

format:
	$(VENV_BIN)ruff check --fix . && $(VENV_BIN)ruff format .

api:             ## http://localhost:8000/docs
	$(VENV_BIN)uvicorn src.api:app --reload

ui:              ## http://localhost:8501
	$(VENV_BIN)streamlit run app/streamlit_app.py

monitor:         ## PSI drift report from the audit log
	$(PY) -m src.monitor

docker:          ## build the image (run `make pipeline` first so models/ and reports/ exist)
	docker build -t credit-decisioning-engine .

compose:         ## api on :8000 and demo on :8501
	docker compose up --build

publish-check:   ## fail if reports/ came from synthetic data (run before git push)
	@$(PY) -c "import json,sys; m=json.load(open('reports/metrics.json')); sys.exit(0 if m['meta']['resume_safe'] else 'reports/ is from SYNTHETIC data: run make pipeline on the real data before publishing')"

all: data pipeline results notebooks test publish-check   ## real-data end to end

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov logs
