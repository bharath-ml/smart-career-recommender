# backend/matcher.py
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import embed_texts, EMBEDDINGS_OK, load_model

_logger = logging.getLogger(__name__)

COURSES_CSV = "data/courses.csv"

def load_courses(path=COURSES_CSV):
    df = pd.read_csv(path)
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # ensure expected columns exist, if missing add defaults
    for col in ["title", "provider", "duration_weeks", "prerequisites", "skill_tags", "level", "link", "cost", "id"]:
        if col not in df.columns:
            df[col] = ""
    # clean some fields
    df['skill_tags'] = df['skill_tags'].fillna("").astype(str)
    df['prerequisites'] = df['prerequisites'].fillna("").astype(str)
    df['title'] = df['title'].fillna("").astype(str)
    df['provider'] = df['provider'].fillna("").astype(str)
    df['link'] = df['link'].fillna("").astype(str)
    # normalize link: ensure http/https
    def fix_link(x):
        x = str(x).strip()
        if x == "" or x.lower().startswith("http"):
            return x
        return "https://" + x
    df['link'] = df['link'].apply(fix_link)
    # ensure duration numeric
    df['duration_weeks'] = pd.to_numeric(df['duration_weeks'], errors='coerce').fillna(0).astype(int)
    # normalize level lower
    df['level'] = df['level'].fillna("").astype(str).str.lower()
    return df

def course_to_text(row):
    # build a richer text for embeddings
    parts = [
        f"Title: {row['title']}",
        f"Provider: {row['provider']}",
        f"Level: {row['level']}",
        f"Skills: {row['skill_tags'].replace('|', ', ')}",
        f"Prereqs: {row['prerequisites'].replace('|', ', ')}",
        f"Duration: {row['duration_weeks']} weeks"
    ]
    return ". ".join([p for p in parts if p])

# load once at import to avoid repeating CSV parsing
_COURSES_DF = load_courses(COURSES_CSV)
_COURSE_TEXTS = [_ for _ in (_COURSES_DF.apply(course_to_text, axis=1).tolist())]
# compute embeddings ONCE
_logger.info("Computing course embeddings (will use model if available)...")
_COURSE_EMBS = embed_texts(_COURSE_TEXTS)
_logger.info("Course embeddings ready (shape: %s).", getattr(_COURSE_EMBS, "shape", None))

# helper functions
def safe_cosine(a, b):
    try:
        sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        if not np.isfinite(sim):
            return 0.0
        return float(sim)
    except Exception:
        return 0.0

def skill_overlap_score(user_skills, course_skill_tags):
    # both inputs should be sets of lowercased tokens
    if not user_skills:
        return 0.0
    course_skills = set([s.strip().lower() for s in course_skill_tags.split("|") if s.strip()])
    if not course_skills:
        return 0.0
    overlap = user_skills.intersection(course_skills)
    # normalized overlap ratio
    return len(overlap) / max(len(course_skills), 1)

def prereq_satisfied(user_skills, prereq_text):
    if not prereq_text or prereq_text.strip().lower() in ["none", ""]:
        return True
    prereqs = set([s.strip().lower() for s in prereq_text.split("|") if s.strip()])
    if not prereqs:
        return True
    return prereqs.issubset(user_skills)  # all prereqs present

# main recommend function
def recommend(user_profile: dict, top_k: int = 10):
    """
    user_profile: dict with keys: education, major, technical_skills (list), soft_skills (list),
                  target_domain (str), preferred_duration_weeks (int)
    """
    # canonicalize user skills
    user_skills = set([s.strip().lower() for s in user_profile.get("technical_skills") or [] if s and str(s).strip()])
    # build user blob
    blob_parts = [
        f"Education: {user_profile.get('education', '')}",
        f"Major: {user_profile.get('major', '')}",
        f"Technical Skills: {', '.join(list(user_skills))}",
        f"Soft Skills: {', '.join(user_profile.get('soft_skills') or [])}",
        f"Target Domain: {user_profile.get('target_domain','')}",
        f"Preferred Duration Weeks: {user_profile.get('preferred_duration_weeks') or ''}"
    ]
    user_blob = ". ".join([p for p in blob_parts if p])
    # embed user
    user_emb = embed_texts([user_blob])[0]

    results = []
    for idx, row in _COURSES_DF.iterrows():
        # semantic similarity if embeddings available
        sem = 0.0
        try:
            sem = safe_cosine(user_emb, _COURSE_EMBS[idx])
        except Exception:
            sem = 0.0

        # skill overlap
        skill_score = skill_overlap_score(user_skills, row['skill_tags'])

        # prereq check
        prereq_ok = prereq_satisfied(user_skills, row['prerequisites'])
        prereq_factor = 1.0 if prereq_ok else 0.6

        # level penalty (do not recommend advanced to beginners)
        level = (row.get('level') or "").lower()
        level_penalty = 0.0
        # user education/major could be used later to infer user level; for now assume beginner if no skills
        if not user_skills and level in ["advanced", "intermediate"]:
            level_penalty = -0.25  # reduce final score
        # duration match score
        preferred_dur = int(user_profile.get("preferred_duration_weeks") or 0)
        dur_score = 1.0
        if preferred_dur > 0:
            # closeness measure
            dur_score = max(0.0, 1.0 - abs(row.get('duration_weeks', 0) - preferred_dur) / max(preferred_dur, 1))

        # combine signals: weights (tweakable)
        # If embeddings model failed entirely, sem will be 0 and skill_score will dominate
        final_raw = (0.65 * sem) + (0.30 * skill_score)  # base
        final_raw = final_raw * prereq_factor
        final_raw = final_raw * (0.8 + 0.2 * dur_score)  # small boost for duration fit
        final_raw = final_raw + level_penalty  # apply penalty
        # normalize to 0..1
        final = max(0.0, min(1.0, final_raw))

        results.append({
            "id": row.get('id'),
            "title": row.get('title'),
            "provider": row.get('provider'),
            "duration_weeks": int(row.get('duration_weeks') or 0),
            "prerequisites": row.get('prerequisites'),
            "skill_tags": row.get('skill_tags'),
            "level": row.get('level'),
            "link": row.get('link'),
            "cost": row.get('cost'),
            "sim": float(sem),
            "skill_overlap": float(skill_score),
            "prereq_ok": bool(prereq_ok),
            "raw_score": float(final_raw),
            "fit_score": int(round(final * 100))
        })

    # sort and take top_k
    results = sorted(results, key=lambda r: r['fit_score'], reverse=True)
    top = results[:top_k]
    return top



