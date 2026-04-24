import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_DB_DIR"] = "/tmp/chroma_store"

import threading
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

recommender = None

def load_model():
    global recommender
    from llm import get_recommendation
    recommender = get_recommendation
    print("Model loaded!", flush=True)

threading.Thread(target=load_model, daemon=True).start()

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
    import asyncio
    waited = 0
    while recommender is None and waited < 300:
        await asyncio.sleep(1)
        waited += 1
    if recommender is None:
        return {"error": "Model still loading, try again in 60 seconds"}
    history_ids = [item.tmdb_id for item in request.history]
    result = recommender(
        preferences=request.preferences,
        history=request.history,
        history_ids=history_ids
    )
    return {"tmdb_id": result["tmdb_id"], "user_id": request.user_id, "description": result["description"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)