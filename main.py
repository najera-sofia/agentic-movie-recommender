# -*- coding: utf-8 -*-
"""
FastAPI wrapper for movie recommender agent
This is the entry point for Leapcell deployment.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json

# Import the recommendation function from llm.py
from llm import get_recommendation

app = FastAPI(title="Movie Recommender Agent")

class HistoryItem(BaseModel):
    tmdb_id: int
    name: str

class RecommendationRequest(BaseModel):
    user_id: int
    preferences: str
    history: List[HistoryItem] = []

class RecommendationResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str

@app.post("/recommend")
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get a movie recommendation based on user preferences and watch history.
    
    Args:
        request: RecommendationRequest with user_id, preferences, and history
        
    Returns:
        RecommendationResponse with tmdb_id, user_id, and description
    """
    # Extract history IDs
    history_ids = [item.tmdb_id for item in request.history]
    
    # Get recommendation from llm.py
    result = get_recommendation(
        preferences=request.preferences,
        history=request.history,
        history_ids=history_ids
    )
    
    # Add user_id to response
    return RecommendationResponse(
        tmdb_id=result["tmdb_id"],
        user_id=request.user_id,
        description=result["description"]
    )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Movie Recommender Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
