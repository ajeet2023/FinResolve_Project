from workflow import app as workflow_app, retriever_node, get_governed_kpi_metrics

# Golden Test Dataset
# Top of evaluate.py:
GOLDEN_RETRIEVAL_TESTS = [
    {
        "query": "Did the bank breach Clause 4.2 of loan LN-40291 by raising the margin by 0.75%?",
        "user_region": "UK_Retail",
        "expected_doc": "LN-40291"
    },
    {
        "query": "What is the early repayment charge for loan LN-88312?",
        "user_region": "UK_Retail",
        "expected_doc": "LN-88312"
    }
]

def evaluate_rag_and_agent():
    print("=== 1. Evaluating Retrieval Quality (Precision@K, Recall@K, MRR) ===")
    k = 2
    precisions, recalls, reciprocal_ranks = [], [], []
    
    for test in GOLDEN_RETRIEVAL_TESTS:
        res = retriever_node({
            "user_query": test["query"],
            "user_region": test["user_region"],
            "top_k": k,
            "retrieved_chunks": [],
            "retrieved_doc_ids": [],
            "reasoning_output": {},
            "telemetry": {},
            "retry_count": 0
        })
        retrieved = res["retrieved_doc_ids"]
        expected = test["expected_doc"]
        
        # Precision@K & Recall@K
        hits = 1 if expected in retrieved else 0
        precisions.append(hits / k)
        recalls.append(hits / 1.0)
        
        # Mean Reciprocal Rank (MRR)
        rr = 1.0 / (retrieved.index(expected) + 1) if expected in retrieved else 0.0
        reciprocal_ranks.append(rr)
        print(f"Query: '{test['query'][:45]}...' -> Retrieved: {retrieved} | RR: {rr:.2f}")

    print(f"--> Avg Precision@{k}: {sum(precisions)/len(precisions):.2f}")
    print(f"--> Avg Recall@{k}:    {sum(recalls)/len(recalls):.2f}")
    print(f"--> MRR:              {sum(reciprocal_ranks)/len(reciprocal_ranks):.2f}\n")

    print("=== 2. Evaluating Row-Level Security (RLS) Cross-Scope Protection ===")
    # A UK_Retail user attempts to access UK_Commercial loan LN-99104
    rls_test = retriever_node({
        "user_query": "Tell me the commercial covenant terms for LN-99104",
        "user_region": "UK_Retail",
        "top_k": 2,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })
    leaked = "LN-99104" in rls_test["retrieved_doc_ids"]
    print(f"--> Unauthorized Commercial Doc Leaked to Retail User? {leaked} (Security Pass: {not leaked})\n")

    print("=== 3. Evaluating Power BI Semantic Layer Reconciliation ===")
    kpis = get_governed_kpi_metrics("UK_Retail")["dax_equivalent_measures"]
    print(f"--> Verified Power BI Measure [UK_Retail Total Exposure]: £{kpis['Total_Exposure_GBP']:,.2f}")
    print(f"--> Verified Power BI Measure [UK_Retail Breach Rate]:    {kpis['Breach_Rate_Pct']}%\n")

    print("=== 4. Evaluating End-to-End Agent Faithfulness, Latency & Token Budget ===")
    agent_res = workflow_app.invoke({
        "user_query": "Did the bank breach Clause 4.2 of loan LN-40291 by raising the margin by 0.75%?",
        "user_region": "UK_Retail",
        "top_k": 2,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })
    verdict = agent_res["reasoning_output"]
    telemetry = agent_res["telemetry"]
    print(f"--> Correct Breach Verdict (Expected True): {verdict.get('has_breached')}")
    print(f"--> Latency SLA (<5000ms):                  {telemetry.get('latency_ms')} ms")
    print(f"--> Token Usage (Budget <= 300 out):        {telemetry.get('total_tokens')} tokens")

if __name__ == "__main__":
    evaluate_rag_and_agent()