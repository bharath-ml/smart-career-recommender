# ui/app.py

import threading
from fastapi import FastAPI
import uvicorn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.main import app as fastapi_app


def run_backend():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_backend, daemon=True)
thread.start()





import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/recommend"

st.set_page_config(page_title="Smart Career Recommender", layout="wide")
st.title("🎯 Smart Career — Course & Certification Recommender")

st.markdown("""
    <style>
    .course-card {
        padding: 16px;
        border-radius: 12px;
        background-color: #ffffff;
        margin-top: 12px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .score {
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Quick filters")
    max_duration = st.slider("Max duration (weeks)", 1, 52, 52)
    only_free = st.checkbox("Only free courses", value=False)
    min_score = st.slider("Minimum fit score", 0, 100, 0)

with st.form("profile"):
    st.subheader("Enter Your Profile")
    education = st.selectbox("Education Level", ["B.Tech", "B.Sc", "M.Tech", "M.Sc", "Diploma", "MCA", "PhD", "Other"])
    major = st.selectbox("Major / Degree", ["Computer Science", "IT", "ECE", "EEE", "Mechanical", "Civil", "Other"])
    tech = st.multiselect("Technical Skills", ["python", "java", "sql", "pandas", "numpy", "machine learning", "deep learning", "react", "docker", "kubernetes", "aws", "gcp", "html", "css", "javascript", "nodejs"])
    soft = st.multiselect("Soft Skills", ["communication", "teamwork", "leadership", "time management", "problem-solving"])
    pref_dur = st.slider("Preferred course duration (weeks)", 1, 52, 12)
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    profile = {
        "education": education,
        "major": major,
        "technical_skills": tech,
        "soft_skills": soft,
        
        "preferred_duration_weeks": pref_dur
    }
    with st.spinner("Querying recommender..."):
        try:
            resp = requests.post(API_URL, json=profile, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                recs = data.get("recommendations", [])
                # apply UI-side filters
                filtered = []
                for r in recs:
                    if r.get("duration_weeks", 0) > max_duration:
                        continue
                    if only_free and str(r.get("cost","")).lower() not in ["free",""]:
                        continue
                    if r.get("fit_score", 0) < min_score:
                        continue
                    filtered.append(r)
                if not filtered:
                    st.warning("No recommendations match your filters. Try relaxing filters.")
                else:
                    st.subheader(f"Top {len(filtered)} Recommendations")
                    for r in filtered:
                        st.markdown(f"""
                            <div class="course-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <h3 style="margin:0;">{r.get('title')}</h3>
                                        <div style="color:#666;">{r.get('provider')} • {r.get('level')} • {r.get('duration_weeks')} weeks</div>
                                    </div>
                                    <div style="text-align:right;">
                                        <div class="score">{r.get('fit_score')}%</div>
                                    </div>
                                </div>
                                <p style="margin-top:8px; margin-bottom:6px;">
                                    <b>Skills:</b> {r.get('skill_tags', '').replace('|', ', ')}<br>
                                    <b>Prereqs:</b> {r.get('prerequisites') or 'None'}
                                </p>
                                <p>
                                    <a href="{r.get('link')}" target="_blank">👉 View Course</a>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"Backend error: {resp.status_code} {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to contact backend: {e}")

