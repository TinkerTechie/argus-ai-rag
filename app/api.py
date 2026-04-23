import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.workflow import build_graph

# Initialize app
app = FastAPI(title="Argus AI API")

# Load graph once
graph = build_graph()

# Request schema
class QueryRequest(BaseModel):
    query: str


# Health check
@app.get("/")
async def home():
    return {"message": "Argus AI is running 🚀"}


# Main endpoint
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Offload the long-running graph execution to a separate thread
    # to avoid blocking the event loop.
    result = await asyncio.to_thread(
        graph.invoke, 
        {"query": request.query, "revision_count": 0}
    )

    return {
        "answer": result["draft_answer"],
        "score": result["critique_score"],
        "feedback": result["critique_feedback"]
    }