# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from backend.matcher import recommend

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("backend")

app = FastAPI(title="Smart Career Recommender")

# allow Streamlit or local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserProfile(BaseModel):
    education: Optional[str] = ""
    major: Optional[str] = ""
    technical_skills: Optional[List[str]] = []
    soft_skills: Optional[List[str]] = []
    target_domain: Optional[str] = ""
    preferred_duration_weeks: Optional[int] = 0

@app.post("/recommend")
def recommend_endpoint(profile: UserProfile, top_k: int = 10):
    _logger.info("Received recommendation request: skills=%s target=%s", profile.technical_skills, profile.target_domain)
    profile_dict = profile.dict()
    recs = recommend(profile_dict, top_k=top_k)
    return {"profile": profile_dict, "recommendations": recs}

