
import os
os.environ["CHROMA_DB_DIR"] = "/tmp/chroma_store"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import threading

app = FastAPI()

# Load model in background AFTER server starts
recommender = None

def load_model():
    global recommender
    from llm import get_recommendation
    recommender = get_recommendation
    print("Model loaded!")

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
async def health2():
    return {"status": "ok"}

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    import asyncio
    waited = 0
    while recommender is None and waited < 120:
        await asyncio.sleep(1)
        waited += 1
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again in 30 seconds")
    
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