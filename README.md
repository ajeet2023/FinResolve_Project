# FinResolve: Enterprise Compliance & Self-Serve Analytics Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-v2-purple.svg)](https://modelcontextprotocol.io/)

**FinResolve** is a production-grade, dual-interface enterprise AI agent designed for automated mortgage contract compliance auditing and self-serve financial analytics. Built and tested on **Python 3.13**, it bridges the gap between unstructured legal documents (PDF agreements) and structured business intelligence (Power BI semantic layers) with strict cost, latency, and security guardrails.

---

## 🏗️ Architecture Overview

```text
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