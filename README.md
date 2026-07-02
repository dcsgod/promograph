# Promograph

**Graph-Steered Conversational Trade Promotion Optimization.**

A fully **open-source, $0** platform that answers *"what happens to network-wide
revenue if I promote this product?"* - combining:

1. **Graph-steered conversational UI** - ask a promo question, watch the relevant
   product subgraph build live, and **click a node to steer the next turn**.
2. **Graph-enhanced cannibalization & halo predictor** - a GNN (GraphSAGE) +
   LightGBM pipeline that predicts **net ROI after cannibalization**, not isolated
   single-SKU lift.
3. A **quant pricing agent** that picks the margin-safe optimal discount depth,
   balancing net margin against inventory-clearance value.

Everything runs locally at no cost. Databricks **Free Edition** Genie Spaces are
supported for the NL-query layer, with a local mock so nothing is blocked.

## Architecture

| Layer | Component | Tech ($0) |
|---|---|---|
| 5 UI | chat + live graph, click-to-steer | React + React Flow + SSE |
| Orchestrator | function-calling loop + node-clicks | **OpenAI SDK → Ollama** |
| 4 Pricing | margin-safe optimal depth | scipy |
| 3 GNN + forecast | SKU embeddings → net ROI | PyTorch Geometric + LightGBM |
| 2 Graph | substitutes / co-purchase / promo | Neo4j **or** DuckDB fallback |
| 1 Data & NL query | governed sales/promo/inventory | Databricks Genie (Free) / mock + DuckDB |

See [docs/architecture.md](docs/architecture.md) and [docs/schema.md](docs/schema.md).

## Quickstart

### 0. Prerequisites
- Python 3.10+ and Node 18+.
- (Optional) [Ollama](https://ollama.com) for conversational mode - without it the
  app runs a deterministic planner that still drives the full pipeline.
- (Optional) Docker for Neo4j - without it the DuckDB graph backend is used.

### 1. Backend setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -e .

# GNN is optional (numpy SVD fallback exists). For real GraphSAGE:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

### 2. Build data → graph → models
```bash
python -m backend.data_gen.generate          # synthetic dunnhumby-shaped data -> DuckDB
python -m backend.graph_store.build_edges     # substitute / co-purchase / promo edges
python -m backend.gnn.train_graphsage         # SKU embeddings -> data/embeddings.parquet
python -m backend.forecasting.train_lgbm      # baseline-vs-promo-lift model
```
(Or `make data && make etl && make train`.)

Smoke-test the whole brain with no server:
```bash
python -m backend.orchestrator.demo
```

### 3. (Optional) Local LLM
```bash
ollama pull qwen2.5:3b     # good tool-calling on CPU; use qwen2.5:7b if you have a GPU
ollama serve               # exposes the OpenAI-compatible endpoint on :11434
```

### 4. Run
```bash
uvicorn backend.api.app:app --port 8001     # backend
cd frontend && npm install && npm run dev   # frontend on http://localhost:5173
```
Open http://localhost:5173, ask *"What's the optimal discount for ground coffee?"*,
then click a red (cannibalized) node to drill in.

## Switching to real Databricks Genie + Neo4j
Copy `.env.example` to `.env` and set:
```
TPO_GENIE_BACKEND=databricks
DATABRICKS_HOST=...   DATABRICKS_TOKEN=...   DATABRICKS_WAREHOUSE_ID=...
GENIE_SPACE_SALES=...  GENIE_SPACE_PROMO=...  GENIE_SPACE_INVENTORY=...
TPO_GRAPH_BACKEND=neo4j
```
Then:
```bash
docker compose up -d neo4j
python data/etl/databricks_to_duckdb.py       # pull dunnhumby from Unity Catalog
python -m backend.graph_store.build_edges
python -m backend.graph_store.etl_duckdb_to_neo4j
```
Load the dunnhumby "Complete Journey" CSVs into Unity Catalog (`catalog.schema`
from `.env`) and build 2-3 Genie Spaces over the `product`, `promo_events`, and
sales tables. Genie free tier is ~5 questions/min - ETL uses the SQL connector,
not Genie.

## How it works (the interesting part)
- **Cannibalization** = substitutes (same sub-commodity) losing full-margin sales
  when the focus SKU is promoted.
- **Halo** = co-purchased neighbours (mined from baskets) gaining sales.
- The forecaster consumes **graph-derived features** (substitute-promo pressure,
  co-purchase pull) + GNN embeddings, so predictions are network-aware.
- The pricing agent maximizes `net_incremental_margin + clearance_value` subject to
  a margin floor - which is why it recommends promoting an overstocked,
  low-cannibalization item but warns against a deep cut on a substitutable one.

## Layout
```
backend/{data_gen,graph_store,gnn,forecasting,pricing_agent,genie_clients,llm,orchestrator,api}
frontend/src/{components/{chat,graph},lib}
data/etl   docs/   docker-compose.yml   pyproject.toml   Makefile
```
