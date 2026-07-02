# Architecture

## End-to-end flow

```
                         ┌──────────────────────────────────────────────┐
                         │  Frontend (React + React Flow)               │
                         │  ChatPanel  ── SSE tokens ─────┐             │
                         │  GraphPanel ◄─ graph_node/edge │  click node │
                         └───────────────┬────────────────┴──────┬──────┘
                                         │ POST /chat            │ POST /node-click
                                         ▼                       ▼
                         ┌──────────────────────────────────────────────┐
                         │  FastAPI + SSE (backend/api)                  │
                         │  streams: token | graph_node | graph_edge |   │
                         │           recommendation | done               │
                         └───────────────┬──────────────────────────────┘
                                         ▼
                         ┌──────────────────────────────────────────────┐
                         │  Orchestrator (OpenAI SDK function-calling)   │
                         │  supervisor prompt → model picks tools        │
                         └───┬───────┬───────────┬────────────┬──────────┘
                             ▼       ▼           ▼            ▼
                       genie_query get_subgraph predict_impact optimize_discount
                             │       │           │            │
                    ┌────────┘   ┌───┘       ┌───┘        ┌───┘
                    ▼            ▼           ▼            ▼
               Genie/DuckDB   Neo4j     GNN emb +     scipy optimizer
               (NL→SQL)      subgraph   LightGBM      (margin-safe depth)
                                        net ROI
```

## Data vs. control flow

* **Data flows up:** Databricks Unity Catalog → (SQL connector) → DuckDB mirror → Neo4j graph → GNN embeddings → LightGBM forecasting → pricing agent.
* **Control flows down:** a user message or a **node click** enters the orchestrator, which calls the relevant tools and re-queries lower layers, then streams a new LLM turn + incremental graph updates back up.

## Why these choices

| Concern | Choice | Rationale |
|---|---|---|
| LLM runtime | Ollama + OpenAI SDK | local, $0, OpenAI-compatible streaming + tool calls |
| Orchestration | function-calling loop | faster & more debuggable than a graph executor; no heavy deps |
| Graph DB | Neo4j Community | mature Cypher + free; treated as a rebuildable cache |
| Embeddings | GraphSAGE (PyG) | inductive, small, CPU-trainable on a modest graph |
| Forecasting | LightGBM | fast on CPU, handles tabular + embedding features well |
| Pricing | scipy.optimize | transparent constrained optimization over a demand curve |
| NL→SQL | Databricks Genie | governed, $0 on Free Edition; mockable locally via DuckDB |

## The Genie seam

`backend/genie_clients/base.py` defines `GenieClient.ask() -> GenieResult`. Two implementations:

* `MockGenieClient` - maps a small set of intents to canned SQL run against the local DuckDB mirror. Default; unblocks all development.
* `DatabricksGenieClient` - real Conversation API: `start-conversation` → poll `messages/{id}` → `query-result`. Selected by `TPO_GENIE_BACKEND=databricks`.

Because the mock returns the exact same `GenieResult` shape, swapping to real Genie is a one-line env change.

## The LLM ↔ Graph contract

Every entity the assistant references is emitted as a **typed SSE event**, not prose:

* `graph_node`  `{id, type, label, metrics}`
* `graph_edge`  `{source, target, type, weight}`
* `recommendation` `{sku, discount_depth, net_roi, margin_ok, rationale}`

The frontend renders the graph purely from these events; it never parses the assistant's natural-language text. A **node click** is posted back as a first-class turn so the model has full context of what was clicked.
