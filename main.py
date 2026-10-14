import traceback
from functools import lru_cache
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import joblib

from workflow import app as workflow_app, retriever_node, get_governed_kpi_metrics

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML Router models...")
    ml_models["router"] = joblib.load("models/router_model.joblib")
    ml_models["vectorizer"] = joblib.load("models/tfidf_vectorizer.joblib")
    yield
    ml_models.clear()

app = FastAPI(
    title="FinResolve Enterprise REST API",
    version="2.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str
    user_region: str = Field(default="UK_Retail", description="RLS scope: UK_Retail, UK_Commercial, or ALL")
    top_k: int = Field(default=2, ge=1, le=5)

# In-memory LRU cache to prevent burning tokens on repeated complex queries
@lru_cache(maxsize=128)
def cached_workflow_execution(user_query: str, user_region: str, top_k: int):
    return workflow_app.invoke({
        "user_query": user_query,
        "user_region": user_region,
        "top_k": top_k,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "router_loaded": "router" in ml_models}

@app.get("/api/v1/powerbi-kpis")
def powerbi_semantic_kpis(user_region: str = Query("UK_Retail", description="RLS Scope")):
    """Returns governed DAX-equivalent metrics so agent numbers always match Power BI."""
    return get_governed_kpi_metrics(user_region)

@app.get("/api/v1/clauses")
def search_clauses(
    search_term: str = Query(...),
    user_region: str = Query("UK_Retail"),
    top_k: int = Query(2)
):
    """Executes Hybrid Search + Row-Level Security without calling the LLM."""
    result = retriever_node({
        "user_query": search_term,
        "user_region": user_region,
        "top_k": top_k,
        "retrieved_chunks": [],
        "retrieved_doc_ids": [],
        "reasoning_output": {},
        "telemetry": {},
        "retry_count": 0
    })
    return {
        "status": "success",
        "rls_region": user_region,
        "retrieved_doc_ids": result["retrieved_doc_ids"],
        "retrieved_chunks": result["retrieved_chunks"]
    }

@app.post("/api/v1/query")
def handle_query(request: QueryRequest):
    vectorized_query = ml_models["vectorizer"].transform([request.query])
    
    # Use predict_proba() for cost-sensitive decision thresholding
    probabilities = ml_models["router"].predict_proba(vectorized_query)[0]
    classes = list(ml_models["router"].classes_)
    simple_prob = float(probabilities[classes.index("simple")])
    complex_prob = float(probabilities[classes.index("complex")])
    
    # Only route to "simple" if the ML model is >= 65% confident it is simple.
    # Otherwise, fail-safe to the "complex" compliance pipeline to avoid False Negatives.
    intent = "simple" if simple_prob >= 0.65 else "complex"
    ml_telemetry = {
        "predicted_intent": intent,
        "simple_probability": round(simple_prob, 4),
        "complex_probability": round(complex_prob, 4),
        "decision_threshold": 0.65
    }
    
    if intent == "simple":
        return {
            "intent": "simple",
            "status": "success",
            "ml_router_metrics": ml_telemetry,
            "rls_region": request.user_region,
            "data": get_governed_kpi_metrics(request.user_region)
        }
        
    elif intent == "complex":
        try:
            result = cached_workflow_execution(request.query, request.user_region, request.top_k)
            return {
                "intent": "complex",
                "status": "success",
                "ml_router_metrics": ml_telemetry,
                "data": result["reasoning_output"],
                "telemetry": result["telemetry"]
            }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))