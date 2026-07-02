# TPO - convenience targets. On Windows use Git Bash, or run the underlying
# `python -m ...` commands directly (see README). `uv run` is used if present.

PY ?= python

.PHONY: help install neo4j-up neo4j-down data etl train serve frontend demo test clean

help:
	@echo "Targets:"
	@echo "  install     Install python deps (core). Add GNN separately (see README)."
	@echo "  neo4j-up    Start Neo4j via docker-compose"
	@echo "  neo4j-down  Stop Neo4j"
	@echo "  data        Generate synthetic dunnhumby-shaped data into DuckDB"
	@echo "  etl         Load DuckDB graph into Neo4j"
	@echo "  train       Train GNN embeddings + LightGBM lift model"
	@echo "  serve       Run FastAPI backend (uvicorn, :8001)"
	@echo "  frontend    Run the Vite dev server (:5173)"
	@echo "  demo        One-shot end-to-end smoke test of the orchestrator"
	@echo "  test        Run pytest"

install:
	pip install -e ".[dev]"

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

data:
	$(PY) -m backend.data_gen.generate

etl:
	$(PY) -m backend.graph_store.etl_duckdb_to_neo4j

train:
	$(PY) -m backend.gnn.train_graphsage
	$(PY) -m backend.forecasting.train_lgbm

serve:
	uvicorn backend.api.app:app --reload --port 8001

frontend:
	cd frontend && npm install && npm run dev

demo:
	$(PY) -m backend.orchestrator.demo

test:
	pytest -q

clean:
	rm -f data/*.duckdb data/*.parquet data/*.txt
