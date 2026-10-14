from mcp.server.mcpserver import MCPServer
from workflow import app as workflow_app, retriever_node, get_governed_kpi_metrics

# Initialize the MCP v2 Server
mcp = MCPServer("FinResolve-Compliance-MCP")

@mcp.tool()
def search_mortgage_clauses(query: str, user_region: str = "UK_Retail") -> list[str]:
    """Hybrid search (pgvector + BM25/FTS) with Row-Level Security filtering by region."""
    res = retriever_node({
        "user_query": query,
        "user_region": user_region,
        "top_k": 2,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })
    return res["retrieved_chunks"]

@mcp.tool()
def get_powerbi_verified_kpis(user_region: str = "UK_Retail") -> dict:
    """Returns governed Power BI semantic KPIs (exposure, breach rate, loan count) for a region."""
    return get_governed_kpi_metrics(user_region)

@mcp.tool()
def evaluate_contract_breach(query: str, user_region: str = "UK_Retail") -> dict:
    """Executes the full LangGraph compliance workflow with RLS and token budget controls."""
    res = workflow_app.invoke({
        "user_query": query,
        "user_region": user_region,
        "top_k": 2,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })
    return {
        "verdict": res["reasoning_output"],
        "telemetry": res["telemetry"]
    }

if __name__ == "__main__":
    mcp.run()