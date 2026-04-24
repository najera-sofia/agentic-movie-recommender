import os
os.environ["CHROMA_DB_DIR"] = "/tmp/chroma_store"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load synchronously at startup - blocks until ready
print("[STARTUP] Loading recommendation engine...", flush=True)
from llm import get_recommendation
print("[STARTUP] Ready!", flush=True)

app = FastAPI()

class HistoryItem(BaseModel):
    tmdb_id: int
    name: str

class RecommendationRequest(BaseModel):
    user_id: int
    preferences: str
    history: List[HistoryItem] = []

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/kaithheathcheck")
async def health():
    return {"status": "ok"}

@app.get("/kaithhealthcheck")
async def health2():
    return {"status": "ok"}

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    history_ids = [item.tmdb_id for item in request.history]
    result = get_recommendation(
        preferences=request.preferences,
        history=request.history,
        history_ids=history_ids
    )
    return {
        "tmdb_id": result["tmdb_id"],
        "user_id": request.user_id,
        "description": result["description"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)