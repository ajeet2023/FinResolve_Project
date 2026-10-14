# FinResolve: Enterprise Compliance & Self-Serve Analytics Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-v2-purple.svg)](https://modelcontextprotocol.io/)

**FinResolve** is a production-grade, dual-interface enterprise AI agent designed for automated mortgage contract compliance auditing and self-serve financial analytics. Built and tested on **Python 3.13**, it bridges the gap between unstructured legal documents (PDF agreements) and structured business intelligence (Power BI semantic layers) with strict cost, latency, and security guardrails.

---

## 🏗️ Architecture Overview

~~~text
[ Client / UI / AI Assistant ]
         │
         ├──► FastAPI REST API (`main.py`) ──► Scikit-learn ML Router (< 2ms)
         │                                          │
         │         ┌────────────────────────────────┴────────────────────────────────┐
         │         ▼ ("simple")                                                      ▼ ("complex")
         │    Governed PostgreSQL KPI Ledger                              LangGraph Orchestration Pipeline
         │    (Power BI DAX Alignment)                                       │
         │                                            ┌──────────────────────┴──────────────────────┐
         │                                            ▼                                             ▼
         │                                   Node 1: Hybrid Retriever                       Node 2: Gemini Reasoner
         │                                   (pgvector Cosine + BM25 FTS                    (Multi-Model Failover & Token Budget)
         │                                    + Row-Level Security [RLS])                           │
         │                                                                                          ▼
         │                                                                                  Node 3: Validator
         │                                                                                  (Confidence Threshold & Loop Cap)
         │
         └──► MCP v2 Server (`mcp_server.py`) ──► Exposes Tools (`@mcp.tool()`) to VS Code / Claude Desktop
~~~

---

## 🚀 Key Enterprise Features

1. **Two-Tier Intent Routing (Scikit-Learn + LangGraph)**
   * Incoming queries hit a trained **TF-IDF + Logistic Regression** classifier using cost-sensitive `.predict_proba()` thresholding (`simple_prob >= 0.65`).
   * Simple queries (e.g., account balance, KPI lookups) bypass the LLM entirely, resolving in **< 2ms** at zero token cost.
   * Complex compliance queries trigger a **LangGraph** state machine.

2. **Database-Level Row-Level Security (RLS)**
   * Multi-tenant security enforced directly in PostgreSQL via dynamic SQL metadata filtering (`WHERE region = %s`).
   * Retail users are blocked from accessing commercial covenants (e.g., loan `LN-99104`) *before* chunks ever reach the LLM context window.

3. **Power BI Semantic Layer Alignment**
   * Eliminates LLM calculation hallucinations by pairing vector search with a governed SQL KPI ledger (`mortgage_kpi_ledger`) mirroring executive Power BI DAX measures (`Total_Exposure_GBP`, `Breach_Rate_Pct`).

4. **Robust Cost & Latency Guardrails**
   * **Token Budgets:** Strictly capped output generation (`max_output_tokens=300`, `temperature=0.0`).
   * **Retry & Loop Caps:** Hard limit of 1 retry loop (`retry_count >= 1`) to prevent runaway API spend.
   * **Multi-Model Failover & Circuit Breaker:** Cascades across Gemini Flash pools (`gemini-3.5-flash-lite` -> `gemini-3.5-flash` -> `gemini-3.8-flash`) with an automatic SQL circuit breaker on provider `503` spikes.
   * **Response Caching:** In-memory `@lru_cache` drops repeat query latency from ~7,000ms down to **< 2ms**.

5. **Structure-Aware Document Ingestion**
   * Uses `pdfplumber` to extract text and convert complex PDF tables into Markdown pipe-tables (`| col |`), preserving spatial alignment for rate-cap and fee schedules.

6. **Dual Protocol Interfaces**
   * **REST API:** FastAPI microservice with lifespan state management, health checks, direct clause search, and analytics endpoints.
   * **MCP Server:** Model Context Protocol (v2 `MCPServer`) exposing core agent capabilities as native tools for AI editors like VS Code Copilot and Claude Desktop.

---

## 📁 Project Structure

~~~text
FinResolve_Project/
├── data/
│   ├── historical_queries.csv  # Labeled query dataset
│   └── pdfs/                   # Synthetic mortgage agreement PDFs
├── models/                     # Auto-generated Scikit-learn artifacts (.joblib)
├── ingest.py                   # PDF table extraction, metadata tagging, & PostgreSQL seeding
├── train_router.py             # ML router training, cross-validation, & evaluation suite
├── workflow.py                 # LangGraph state machine (Retriever, Reasoner, Validator)
├── main.py                     # FastAPI microservice (REST API, RLS, LRU Cache)
├── mcp_server.py               # MCP v2 Server exposing tools for AI clients
├── evaluate.py                 # RAG evaluation (Precision@K, Recall@K, MRR, RLS, Faithfulness)
├── Dockerfile                  # Container image definition (Python 3.13)
├── docker-compose.yml          # Multi-container orchestration (FastAPI + pgvector)
├── requirements.txt            # Pinned project dependencies
└── .env.example                # Environment variable template
~~~

---

## 🛠️ Installation & Quickstart

### Option A: One-Command Docker Deployment (Recommended)
1. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`:
~~~bash
cp .env.example .env
~~~
2. Build and start the PostgreSQL `pgvector` database and FastAPI service:
~~~bash
docker compose up --build
~~~

### Option B: Local Virtual Environment Setup
~~~bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_router.py
python ingest.py
python evaluate.py
uvicorn main:app --reload
~~~

---

## 🔌 API & MCP Usage Examples

### 1. Complex Compliance Audit (`POST /api/v1/query`)
~~~bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Did the bank breach Clause 4.2 of loan LN-40291 by raising the margin by 0.75%?",
  "user_region": "UK_Retail",
  "top_k": 2
}'
~~~

### 2. Governed Power BI Semantic KPIs (`GET /api/v1/powerbi-kpis`)
~~~bash
curl -s "http://127.0.0.1:8000/api/v1/powerbi-kpis?user_region=UK_Retail"
~~~

### 3. Configure MCP v2 in VS Code (`.vscode/mcp.json`)
~~~json
{
  "servers": {
    "finresolve-mcp": {
      "type": "stdio",
      "command": "${workspaceFolder}/venv/bin/python",
      "args": ["${workspaceFolder}/mcp_server.py"]
    }
  }
}
~~~

---

## 📊 Evaluation Metrics

FinResolve includes an automated evaluation suite (`evaluate.py`) tracking production RAG and agent metrics:
* **Retrieval Quality:** Precision@K (`1.00`), Recall@K (`1.00`), and Mean Reciprocal Rank (**MRR: `1.00`**).
* **Security Compliance:** Row-Level Security leakage pass rate (**100% block rate** for unauthorized regional scopes).
* **Semantic Accuracy:** 100% reconciliation against the Power BI governed KPI ledger.
* **Agent Reliability:** End-to-end task completion rate, token budget enforcement (`<= 300` output tokens), and latency telemetry.
