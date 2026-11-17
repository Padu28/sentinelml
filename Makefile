.PHONY: install dev pipeline quick api dashboard test lint docker clean

install:        ## Install runtime dependencies (editable)
	pip install -r requirements.txt && pip install -e .

dev:            ## Install dev tooling
	pip install -r requirements-dev.txt && pip install -e .

pipeline:       ## Full experiment: 4 drift types × 5 seeds × 3 policies
	python -m sentinelml.cli pipeline

quick:          ## Quick experiment (2 seeds)
	python -m sentinelml.cli pipeline --quick

api:            ## FastAPI on :8000 (docs at /docs)
	uvicorn api.main:app --reload --port 8000

dashboard:      ## Streamlit dashboard on :8501
	streamlit run dashboard/app.py

test:
	pytest --cov=sentinelml --cov-report=term-missing

lint:
	ruff check src api dashboard tests && ruff format --check src api dashboard tests

docker:
	docker build -f docker/Dockerfile -t sentinelml:latest . && docker run -p 8000:8000 -p 8501:8501 sentinelml:latest

clean:
	rm -rf data/processed/*.db models/*.joblib .pytest_cache .ruff_cache
