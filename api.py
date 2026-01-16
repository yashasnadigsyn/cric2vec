"""
FastAPI backend for Cric2Vec Playground.
Exposes endpoints for player search, matchup simulation, and bunny finder.
"""

from typing import List, Optional
from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy import to avoid loading model at module import time
_engine = None

def get_engine():
    """Lazy initialization of inference engine."""
    global _engine
    if _engine is None:
        from inference import CricInsights
        _engine = CricInsights()
    return _engine


app = FastAPI(
    title="Cric2Vec API",
    description="API for cricket player embedding analysis",
    version="1.0.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class SimulateRequest(BaseModel):
    batter: str
    bowler: str

class SimulateResponse(BaseModel):
    batter: str
    bowler: str
    balls: List[str]
    total_runs: int
    wickets: int

class BunnyRequest(BaseModel):
    bowler: str
    top_k: int = 5

class BunnyResult(BaseModel):
    batter: str
    wicket_prob: float

class PlayerMatch(BaseModel):
    name: str
    score: float


# --- Endpoints ---

@app.get("/api/players", response_model=List[str])
def get_all_players():
    """Get list of all player names for autocomplete."""
    engine = get_engine()
    return sorted(engine.player_to_id.keys())


@app.get("/api/search", response_model=List[PlayerMatch])
def search_players(q: str = Query(..., min_length=1)):
    """
    Search for players by partial name.
    Uses fuzzy matching to find similar names.
    """
    engine = get_engine()
    matches = engine.find_player(q, top_k=10, threshold=50)
    return [PlayerMatch(name=name, score=score) for name, score in matches]


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate_over(req: SimulateRequest):
    """
    Simulate an over between batter and bowler.
    Returns ball-by-ball outcomes.
    """
    engine = get_engine()
    
    try:
        probs_dict = engine.get_matchup_probs(req.batter, req.bowler)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    outcomes = list(probs_dict.keys())
    probabilities = list(probs_dict.values())
    
    balls = random.choices(outcomes, weights=probabilities, k=6)
    
    total_runs = 0
    wickets = 0
    for event in balls:
        if event.startswith('w_'):
            wickets += 1
        elif event == '1_run':
            total_runs += 1
        elif event == '2_3_run':
            total_runs += 2
        elif event == '4_run':
            total_runs += 4
        elif event == '6_run':
            total_runs += 6
        elif event == 'extras':
            total_runs += 1
    
    return SimulateResponse(
        batter=req.batter,
        bowler=req.bowler,
        balls=balls,
        total_runs=total_runs,
        wickets=wickets
    )


@app.post("/api/bunnies", response_model=List[BunnyResult])
def find_bunnies(req: BunnyRequest):
    """
    Find batters most vulnerable to a specific bowler.
    Returns top batters with highest wicket probability.
    """
    engine = get_engine()
    
    try:
        results = engine.find_bunnies(req.bowler, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return [BunnyResult(batter=name, wicket_prob=round(prob, 4)) for name, prob in results]


@app.get("/api/matchup")
def get_matchup(batter: str, bowler: str):
    """
    Get probability distribution for a specific matchup.
    """
    engine = get_engine()
    
    try:
        probs = engine.get_matchup_probs(batter, bowler)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return {
        "batter": batter,
        "bowler": bowler,
        "probabilities": {k: round(v, 4) for k, v in probs.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
