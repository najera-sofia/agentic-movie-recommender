# -*- coding: utf-8 -*-
"""
Movie Recommendation Agent

Environment variables required:
  OLLAMA_API_KEY  - provided by the grader at runtime

DO NOT change the model from gemma4:31b-cloud
"""

import os
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

import json
import re
import time
import argparse
import threading
from functools import lru_cache

import numpy as np
import ollama
import pandas as pd
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
MODEL = "gemma4:31b-cloud"
LLM_TIMEOUT_SECONDS = 9    # stage 1 cap
STAGE2_TIMEOUT_SECONDS = 7  # stage 2 cap: no retry on stage 2
                            # total budget: 12 + 8 = 20s, within the 20s limit
MIN_VOTE_COUNT = 500        # filter out obscure/low-quality movies
CHROMA_PATH = os.environ.get("CHROMA_DB_DIR", os.path.join(os.path.dirname(__file__), ".chroma_store"))
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")

_df = pd.read_csv(DATA_PATH)
_df["tmdb_id"] = _df["tmdb_id"].astype(int)
_df["_score"] = _df["vote_average"] * (
    _df["vote_count"] / (_df["vote_count"] + 500)
) + 6.5 * (500 / (_df["vote_count"] + 500))

# TOP_MOVIES is the full dataset - test.py imports this to build its VALID_IDS set
TOP_MOVIES = _df.copy()
VALID_IDS = set(_df["tmdb_id"].tolist())

# Pre-load embedding model at startup so first request isn't slow
print("[INFO] Loading embedding model...", end=" ", flush=True)
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("ready.")

# Pre-built O(1) lookup dict - avoids expensive _df[_df["tmdb_id"]==tid] calls in hot loops
_movie = {
    int(row["tmdb_id"]): {
        "vote_count":         int(row.get("vote_count") or 0),
        "vote_average":       float(row.get("vote_average") or 0),
        "runtime_min":        int(row["runtime_min"]) if str(row.get("runtime_min","")).replace(".","").isdigit() else 999,
        "year":               int(row["year"]) if str(row.get("year","")).replace(".","").isdigit() else 0,
        "genres":             str(row.get("genres") or ""),
        "original_language":  str(row.get("original_language") or ""),
        "keywords":           str(row.get("keywords") or "").lower(),
        "overview":           str(row.get("overview") or "").lower(),
    }
    for _, row in _df.iterrows()
}

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

EXPANSION_MAP = {
    "superhero":   "superhero comic marvel dc avengers batman action",
    "sci-fi":      "science fiction space future technology dystopia",
    "scifi":       "science fiction space future technology dystopia",
    "rom-com":     "romantic comedy love funny relationship dating",
    "romcom":      "romantic comedy love funny relationship dating",
    "feel-good":   "feel-good uplifting heartwarming comedy family",
    "thriller":    "thriller suspense tension mystery crime",
    "horror":      "horror scary monster ghost supernatural fear",
    "action":      "action adventure fight chase explosion",
    "drama":       "drama emotional powerful story character",
    "animated":    "animation animated cartoon family kids",
    "animation":   "animation animated cartoon family kids",
    "war":         "war military battle historical conflict soldiers",
    "western":     "western cowboy frontier outlaw sheriff",
    "musical":     "musical music singing dancing songs",
    "mystery":     "mystery detective crime whodunit",
    "fantasy":     "fantasy magic dragon medieval world-building",
    "heist":       "heist robbery crime plan getaway",
}

GENRE_NEGATION_MAP = {
    "superhero": ["Action", "Science Fiction", "Adventure"],
    "romance":   ["Romance"],
    "horror":    ["Horror"],
    "thriller":  ["Thriller"],
    "comedy":    ["Comedy"],
    "action":    ["Action"],
    "drama":     ["Drama"],
    "animated":  ["Animation"],
    # "violence" and "gore" both ban action, horror, thriller, war
    "violence":  ["Action", "Horror", "Thriller", "War", "Crime"],
    "gore":      ["Horror"],
    "scary":     ["Horror", "Thriller"],
    "violent":   ["Action", "Horror", "Thriller", "War", "Crime"],
}

NEGATION_PATTERNS = [
    r"\bno\b\s+\w*\s*{kw}",
    r"\bnot? into\b\s+\w*\s*{kw}",
    r"\bavoid\b\s+\w*\s*{kw}",
    r"\bwithout\b\s+\w*\s*{kw}",
    r"\btired of\b\s+\w*\s*{kw}",
    r"\bfed up.{{0,15}}{kw}",
    r"\bsick of.{{0,15}}{kw}",
    r"\b{kw} fatigue\b",
    r"\bno.{{0,5}}{kw}",
]



# ---------------------------------------------------------------------------
# Non-literal / sarcastic query translation
# ---------------------------------------------------------------------------

# Maps non-literal phrases to what the user actually wants
LITERAL_TRANSLATIONS = {
    "so bad it's good":      "cult classic campy B-movie cheap cheesy absurd low-budget unintentional comedy",
    "so bad its good":       "cult classic campy B-movie cheap cheesy absurd low-budget unintentional comedy",
    "guilty pleasure":       "fun entertaining light crowd-pleaser",
    "so bad it is good":     "cult classic campy B-movie cheap cheesy absurd low-budget unintentional comedy",
    "campy":                 "cult classic campy B-movie cheap cheesy absurd low-budget",
    "cult classic":          "cult classic underground obscure campy midnight movie",
    "b-movie":               "B-movie low-budget cult campy cheesy absurd",
    "fall asleep to":        "slow calm peaceful gentle atmospheric",
    "background movie":      "easy light undemanding comfortable",
    "cry my eyes out":       "emotional tearjerker sad moving drama",
    "good cry":              "emotional tearjerker moving heartfelt",
    "turn my brain off":     "fun action spectacle easy entertaining",
    "don't have to think":   "fun action spectacle easy entertaining",
    "doesn't make me think": "fun entertaining light easy-going",
    "impress a date":        "romantic stylish acclaimed crowd-pleaser",
    "look smart":            "acclaimed intellectual critically praised",
    "pretentious":           "acclaimed arthouse intellectual drama",
    "hidden gem":            "underrated cult acclaimed lesser-known",
    "nostalgia":             "classic beloved feel-good familiar comfort",
    "comfort movie":         "warm familiar feel-good comfort classic",
    "rainy day":             "cozy warm feel-good comfort easy",
    "road trip":             "adventure fun energetic entertaining",
    "game night":            "fun group comedy entertaining crowd-pleaser",
    "halloween":             "horror scary spooky thriller suspense",
    "date night":            "romantic feel-good entertaining stylish",
    "grandma":               "gentle warm feel-good family comedy drama classic",
    "grandmother":           "gentle warm feel-good family comedy drama classic",
    "kids":                  "family animation fun heartwarming kids",
    "children":              "family animation fun heartwarming kids",
    "wholesome":             "wholesome feel-good uplifting family warm",
    "brain off":             "fun action spectacle easy entertaining",
    "visually stunning":     "visually spectacular cinematography beautiful stunning",
    "eye candy":             "visually spectacular cinematography beautiful stunning",
    "not in the mood":       "easy light comfortable undemanding",
    "short film":            "short tight paced concise punchy under 90 minutes",
    "under 90":              "short tight paced concise punchy quick",
    "quick watch":           "short easy light fun entertaining fast-paced",
    "documentary":           "documentary true story real events factual non-fiction",
    "documentaries":         "documentary true story real events factual non-fiction",
    "docuseries":            "documentary true story real events factual",
    "true story":            "true story based on real events biographical documentary",
    "based on true":         "true story based on real events biographical",
    "teenager":              "teen coming-of-age young adult adventure action",
    "teenage":               "teen coming-of-age young adult adventure action",
    "teen":                  "teen coming-of-age young adult adventure action",
    "kids will love":        "family animation adventure fun kids young",
    "watch together":        "family feel-good crowd-pleaser entertaining",
    "movie night":           "crowd-pleaser entertaining feel-good popular",
    "with friends":          "crowd-pleaser fun entertaining group comedy action",
    "first date":            "romantic stylish entertaining not too heavy",
    "rainy sunday":          "cozy warm comfort feel-good easy",
    "surprise me":           "acclaimed unique original unexpected critically praised",
    "seen everything":       "acclaimed unique original unexpected critically praised",
}


def _translate_query(preferences: str) -> str:
    """Detect non-literal phrases and translate them to concrete search terms."""
    text = preferences.lower()
    translations = []
    for phrase, meaning in LITERAL_TRANSLATIONS.items():
        if phrase in text:
            translations.append(meaning)
    if translations:
        return preferences + " " + " ".join(translations)
    return preferences


def _safe(v, limit=0):
    s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
    return s[:limit] if limit else s


# Map of well-known movies to their thematic search terms
# Used when user says "like X" to enrich the query with actual themes
MOVIE_THEME_MAP = {
    "parasite":        "class struggle social satire dark thriller rich poor family infiltration",
    "interstellar":    "space exploration epic sci-fi time physics emotion father daughter",
    "inception":       "mind-bending dreams heist layers reality psychological",
    "the dark knight": "psychological crime moral ambiguity villain heroism",
    "get out":         "social horror psychological race satire suburban",
    "hereditary":      "family trauma psychological horror grief supernatural",
    "gone girl":       "psychological thriller marriage deception media",
    "whiplash":        "obsession ambition music psychological intensity",
    "la la land":      "romantic musical dreams ambition bittersweet",
    "knives out":      "whodunit mystery ensemble witty subversive",
    "everything everywhere": "multiverse absurd family love existential comedy",
}


def _expand_query(preferences):
    # First translate any non-literal/sarcastic phrasing
    preferences = _translate_query(preferences)
    text = preferences.lower()
    extras = [exp for key, exp in EXPANSION_MAP.items() if key in text]

    # Enrich "like X" queries with known thematic terms
    for movie_key, themes in MOVIE_THEME_MAP.items():
        if movie_key in text and ("like" in text or "similar" in text or "same" in text):
            extras.append(themes)
            break

    return preferences + (" " + " ".join(extras) if extras else "")


def _detect_negative_genres(preferences):
    text = preferences.lower()
    banned = set()
    for keyword, genres in GENRE_NEGATION_MAP.items():
        for pat in NEGATION_PATTERNS:
            if re.search(pat.format(kw=re.escape(keyword)), text):
                banned.update(genres)
                break
    return banned




# ---------------------------------------------------------------------------
# Language exclusion detection
# ---------------------------------------------------------------------------

LANGUAGE_CODES = {
    "korean":     "ko",
    "japanese":   "ja",
    "french":     "fr",
    "spanish":    "es",
    "chinese":    "zh",
    "hindi":      "hi",
    "german":     "de",
    "italian":    "it",
    "portuguese": "pt",
    "russian":    "ru",
    "thai":       "th",
    "turkish":    "tr",
}

LANGUAGE_NEGATION_PATTERNS = [
    r"\bnot\b.{{0,15}}{lang}",
    r"\bno\b.{{0,10}}{lang}",
    r"\bwithout\b.{{0,10}}{lang}",
    r"\bavoid\b.{{0,10}}{lang}",
    r"\bbut not\b.{{0,10}}{lang}",
    r"\bnon-{lang}\b",
    r"\bexcept.{{0,10}}{lang}",
]


def _detect_excluded_languages(preferences: str) -> set:
    """Return set of ISO language codes the user wants to exclude."""
    text = preferences.lower()
    excluded = set()
    for lang_name, lang_code in LANGUAGE_CODES.items():
        for pat in LANGUAGE_NEGATION_PATTERNS:
            if re.search(pat.format(lang=re.escape(lang_name)), text):
                excluded.add(lang_code)
                break
    return excluded



# ---------------------------------------------------------------------------
# Runtime limit detection
# ---------------------------------------------------------------------------

RUNTIME_PATTERNS = [
    (r'under\s+(\d+)\s*(?:min|minute|minutes|mins|hr|hour|hours)?', 1.0),
    (r'less\s+than\s+(\d+)\s*(?:min|minute|minutes|mins|hr|hour|hours)?', 1.0),
    (r'no\s+(?:more\s+than|longer\s+than)\s+(\d+)\s*(?:min|minute|minutes|mins)?', 1.0),
    (r'(\d+)\s*(?:min|minute|minutes|mins)\s+(?:or\s+less|max|maximum|tops)', 1.0),
    (r'short', None),   # generic "short" - apply a default cap
    (r'quick', None),
]

def _detect_runtime_limit(preferences: str):
    """Return max runtime in minutes, or None if no limit specified."""
    text = preferences.lower()
    for pat, multiplier in RUNTIME_PATTERNS:
        m = re.search(pat, text)
        if m:
            if multiplier is None:
                return 100  # generic "short" -> cap at 100 min
            val = int(m.group(1))
            # If hours mentioned, convert
            if 'hr' in pat or 'hour' in pat:
                if any(w in text[max(0, m.start()-5):m.end()+10] for w in ['hr','hour']):
                    val = val * 60
            return val
    return None



# ---------------------------------------------------------------------------
# Content keyword filtering (catches thematic mismatches genre tags miss)
# ---------------------------------------------------------------------------

CONTENT_KEYWORD_BANS = {
    # phrases that trigger the ban -> keywords/overview terms to exclude
    "peaceful":    {"vampire", "zombie", "monster", "demon", "ghost", "witch", "supernatural", "killer", "murder", "gore"},
    "no violence": {"vampire", "zombie", "monster", "demon", "ghost", "witch", "supernatural", "killer", "murder", "gore"},
    "not scary":   {"vampire", "zombie", "monster", "demon", "ghost", "witch", "supernatural", "horror"},
    "grandma":     {"vampire", "zombie", "monster", "demon", "ghost", "witch", "supernatural", "horror", "murder", "gore"},
    "kids":        {"murder", "gore", "torture", "drug", "sex", "violence"},
    "children":    {"murder", "gore", "torture", "drug", "sex", "violence"},
}


def _detect_banned_keywords(preferences: str) -> set:
    """Return content keywords to ban from overview/keywords fields."""
    text = preferences.lower()
    banned = set()
    for trigger, keywords in CONTENT_KEYWORD_BANS.items():
        if trigger in text:
            banned.update(keywords)
    return banned

# ---------------------------------------------------------------------------
# Step 1: ChromaDB with local sentence-transformers embeddings
# ---------------------------------------------------------------------------

class LocalEmbeddingFunction(EmbeddingFunction):
    """Uses pre-loaded module-level model - no lazy init overhead."""

    def __call__(self, input: Documents) -> Embeddings:
        vectors = _EMBED_MODEL.encode(list(input), show_progress_bar=False)
        return vectors.tolist()


def _build_doc(row):
    """Build a rich text document for embedding from all available metadata."""
    parts = [
        _safe(row.get("title")),
        _safe(row.get("genres")),
        _safe(row.get("overview"), 400),
        _safe(row.get("tagline")),
        _safe(row.get("director")),
        _safe(row.get("top_cast")),
        _safe(row.get("keywords"), 200),
    ]
    return " ".join(p for p in parts if p)


@lru_cache(maxsize=1)
def _get_collection():
    """Return ChromaDB collection, building and persisting it on first call."""
    ef = LocalEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll_name = "movies_v2"
    existing = [c.name for c in client.list_collections()]

    if coll_name in existing:
        return client.get_collection(coll_name, embedding_function=ef)

    print("[INFO] Building ChromaDB index (one-time setup, ~1 min)...")
    collection = client.create_collection(
        coll_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    records = _df.to_dict("records")
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        collection.add(
            documents=[_build_doc(r) for r in batch],
            ids=[str(r["tmdb_id"]) for r in batch],
            metadatas=[
                {
                    "tmdb_id":    int(r["tmdb_id"]),
                    "title":      _safe(r.get("title")),
                    "genres":     _safe(r.get("genres")),
                    "director":   _safe(r.get("director")),
                    "top_cast":   _safe(r.get("top_cast")),
                    "score":      float(r["_score"]),
                    "vote_count": int(r.get("vote_count") or 0),
                }
                for r in batch
            ],
        )
        print(f"  Indexed {min(i + batch_size, len(records))}/{len(records)}", end="\r")
    print("\n[INFO] ChromaDB index ready.")
    return collection


# ---------------------------------------------------------------------------
# Step 2: Filtering and ranking
# ---------------------------------------------------------------------------

def _get_candidates(preferences, history_ids, n=25, banned_keywords=None):
    """Embed the query, retrieve semantically similar movies, apply hard filters."""
    query = _expand_query(preferences)
    banned_genres = _detect_negative_genres(preferences)

    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=min(80, len(_df)))
    metas = results["metadatas"][0]

    excluded_langs = _detect_excluded_languages(preferences)
    max_runtime = _detect_runtime_limit(preferences)

    kept_ids = []
    for meta in metas:
        tid = int(meta["tmdb_id"])
        if tid in history_ids:
            continue
        # Hard filter: skip low-vote-count movies
        if _movie.get(tid, {}).get("vote_count", 0) < MIN_VOTE_COUNT:
            continue
        # Hard filter: skip banned genres
        if banned_genres and any(bg in _safe(meta.get("genres")) for bg in banned_genres):
            continue
        # Hard filter: runtime limit
        if max_runtime is not None:
            m = _movie.get(tid, {})
            if m.get("runtime_min", 999) > max_runtime or m.get("year", 0) >= 2025:
                continue
        # Hard filter: excluded languages
        if excluded_langs and _movie.get(tid, {}).get("original_language") in excluded_langs:
            continue
        # Hard filter: content keyword ban
        if banned_keywords:
            m = _movie.get(tid, {})
            combined = m.get("keywords", "") + " " + m.get("overview", "")
            if any(kw in combined for kw in banned_keywords):
                continue
        kept_ids.append(tid)
        if len(kept_ids) >= n:
            break

    if not kept_ids:
        # Relax vote filter but keep genre, language, runtime, keyword filters
        for meta in metas:
            tid = int(meta["tmdb_id"])
            if tid in history_ids:
                continue
            if banned_genres and any(bg in _safe(meta.get("genres")) for bg in banned_genres):
                continue
            if max_runtime is not None:
                m = _movie.get(tid, {})
                if m.get("runtime_min", 999) > max_runtime or m.get("year", 0) >= 2025:
                    continue
            if excluded_langs and _movie.get(tid, {}).get("original_language") in excluded_langs:
                continue
            kept_ids.append(tid)
            if len(kept_ids) >= n:
                break

    if not kept_ids:
        # Over-constrained: relax genre/language filters but keep runtime and history
        for meta in metas:
            tid = int(meta["tmdb_id"])
            if tid in history_ids:
                continue
            if max_runtime is not None:
                m = _movie.get(tid, {})
                if m.get("runtime_min", 999) > max_runtime or m.get("year", 0) >= 2025:
                    continue
            kept_ids.append(tid)
            if len(kept_ids) >= n:
                break

    if not kept_ids:
        return pd.DataFrame()

    rows = _df[_df["tmdb_id"].isin(kept_ids)].copy()
    order = {tid: i for i, tid in enumerate(kept_ids)}
    rows["_sem_rank"] = rows["tmdb_id"].map(order)
    return rows.sort_values("_sem_rank").reset_index(drop=True)


def _boost_candidates(candidates, history_ids):
    """Soft boost for candidates sharing director/cast with watch history."""
    if candidates.empty or not history_ids:
        return candidates

    hist = _df[_df["tmdb_id"].isin(history_ids)]
    fav_directors, fav_cast = set(), set()
    for _, r in hist.iterrows():
        d = _safe(r.get("director")).strip().lower()
        if d:
            fav_directors.add(d)
        for a in _safe(r.get("top_cast")).split(","):
            a = a.strip().lower()
            if a:
                fav_cast.add(a)

    boosts = []
    for _, row in candidates.iterrows():
        b = 0.0
        if _safe(row.get("director")).lower() in fav_directors:
            b += 0.25
        cast_str = _safe(row.get("top_cast")).lower()
        b += 0.1 * sum(1 for a in fav_cast if a and a in cast_str)
        boosts.append(b)

    candidates = candidates.copy()
    candidates["_boost"] = boosts
    candidates["_final_rank"] = candidates["_sem_rank"] - candidates["_boost"] * 3
    return candidates.sort_values("_final_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: Two-stage LLM
# ---------------------------------------------------------------------------

def _call_with_timeout(fn, timeout):
    result, exc = [None], [None]

    def _run():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"LLM exceeded {timeout}s")
    if exc[0]:
        raise exc[0]
    return result[0]


def _call_with_retry(fn, timeout, retries=1, wait=2.0):
    """Call fn with timeout; retry once after a short wait on failure."""
    last_exc = None
    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(wait)
        try:
            return _call_with_timeout(fn, timeout)
        except Exception as e:
            last_exc = e
            print(f"[WARN] LLM attempt {attempt + 1} failed ({type(e).__name__}), {'retrying' if attempt < retries else 'giving up'}")
    raise last_exc


SELECTION_PROMPT_TEMPLATE = """You are a movie recommendation expert.

The user wants: "{preferences}"
Do NOT recommend these (already seen): {history_text}

Choose the single best movie from the list below. You MUST use a tmdb_id from this exact list.

{movie_list}

Respond with valid JSON only, no markdown, no explanation:
{{"tmdb_id": <integer>, "reason": "<brief rationale>"}}"""

PITCH_PROMPT_TEMPLATE = """You are a passionate film critic recommending a movie to a friend.

Your friend wants: "{preferences}"
Movie to recommend: "{title}" ({year})
Director: {director} | Stars: {cast}
Tagline: {tagline}
Overview: {overview}

Write a persuasive, personal pitch - not a plot summary. Lead with a hook. Name the director or a star. Make them feel why this is the right movie for them right now. {mood_hint} Keep it under 300 characters. Be punchy - 2 to 3 sentences max.

Respond with valid JSON only, no markdown:
{{"description": "<your pitch here>"}}"""


def _detect_mood(preferences):
    pl = preferences.lower()
    if any(w in pl for w in ["funny", "light", "feel-good", "laugh", "comedy"]):
        return "Keep the tone warm, fun, and upbeat."
    elif any(w in pl for w in ["dark", "intense", "gritty", "serious"]):
        return "Use a gripping, serious tone."
    elif any(w in pl for w in ["thought", "smart", "complex", "intelligent"]):
        return "Emphasise intellectual depth and originality."
    return "Match the emotional tone of what the user is asking for."


def _two_stage_llm(preferences, history, candidates, history_id_set):
    api_key = os.environ["OLLAMA_API_KEY"].strip()
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    candidate_ids = set(candidates["tmdb_id"].tolist())
    # Top 5 only, no overviews - keeps stage 1 prompt small for fast API response
    top5 = candidates.head(5)
    movie_list = "\n".join(
        f'tmdb_id={int(r.tmdb_id)} | "{r.title}" ({_safe(r.year)}) | {_safe(r.genres)} | dir: {_safe(r.director) or "N/A"}'
        for r in top5.itertuples()
    )
    history_text = ", ".join(f'"{h}"' for h in history) if history else "none"

    # Stage 1: select best candidate (up to 12s)
    select_prompt = SELECTION_PROMPT_TEMPLATE.format(
        preferences=preferences,
        history_text=history_text,
        movie_list=movie_list,
    )
    r1 = _call_with_retry(
        lambda: client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": select_prompt}],
            format="json",
        ),
        timeout=LLM_TIMEOUT_SECONDS,
        retries=1,
        wait=1.0,
    )
    chosen_id = int(json.loads(r1.message.content)["tmdb_id"])

    # Must be from our filtered candidate list, not history
    if chosen_id not in candidate_ids or chosen_id in history_id_set:
        chosen_id = int(candidates.iloc[0]["tmdb_id"])

    chosen = candidates[candidates["tmdb_id"] == chosen_id].iloc[0]

    # Stage 2: write persuasive pitch (up to 7s, keeping total under 20s)
    pitch_prompt = PITCH_PROMPT_TEMPLATE.format(
        preferences=preferences,
        title=chosen["title"],
        year=_safe(chosen.get("year")),
        director=_safe(chosen.get("director")) or "unknown",
        cast=_safe(chosen.get("top_cast")).split(",")[0].strip() or "ensemble",
        tagline=_safe(chosen.get("tagline")) or "none",
        overview=_safe(chosen.get("overview"))[:180],
        mood_hint=_detect_mood(preferences),
    )
    r2 = _call_with_timeout(
        lambda: client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": pitch_prompt}],
            format="json",
        ),
        timeout=STAGE2_TIMEOUT_SECONDS,
    )
    description = str(json.loads(r2.message.content).get("description", "")).strip()[:300]
    return {"tmdb_id": chosen_id, "description": description}


# ---------------------------------------------------------------------------
# Step 4: Deterministic fallback
# ---------------------------------------------------------------------------

def _fallback(candidates, history_ids, banned_genres=None, excluded_langs=None, max_runtime=None, banned_keywords=None):
    """Return the best candidate deterministically, respecting all active filters."""
    if banned_genres is None:
        banned_genres = set()
    if banned_keywords is None:
        banned_keywords = set()
    for _, row in candidates.iterrows():
        tid = int(row["tmdb_id"])
        if tid in history_ids:
            continue
        # Apply quality floor and genre ban in fallback too
        if int(row.get("vote_count", 0)) < MIN_VOTE_COUNT:
            continue
        if float(row.get("vote_average", 0)) < 5.5:
            continue
        if banned_genres and any(bg in _safe(row.get("genres")) for bg in banned_genres):
            continue
        if excluded_langs and row.get("original_language") in excluded_langs:
            continue
        if max_runtime is not None:
            m = _movie.get(int(row.get("tmdb_id", 0)), {})
            if m.get("runtime_min", 999) > max_runtime or m.get("year", 0) >= 2025:
                continue
        if banned_keywords:
            m = _movie.get(int(row.get("tmdb_id", 0)), {})
            combined = m.get("keywords", "") + " " + m.get("overview", "")
            if any(kw in combined for kw in banned_keywords):
                continue
        director = _safe(row.get("director"))
        cast = _safe(row.get("top_cast")).split(",")[0].strip()
        genres = _safe(row.get("genres"))
        # Build a readable fallback description - still better than a data dump
        year = _safe(row.get("year"))
        rating = f"{row['vote_average']:.1f}"
        if director and cast:
            desc = f'Directed by {director} and starring {cast}, "{row["title"]}" ({year}) is a highly-rated {genres.lower()} that comes strongly recommended. Rated {rating}/10 by fans worldwide.'
        elif director:
            desc = f'"{row["title"]}" ({year}) is a {genres.lower()} from director {director} - one of the most acclaimed films in its genre. Rated {rating}/10.'
        else:
            desc = f'"{row["title"]}" ({year}) is one of the most acclaimed {genres.lower()} films available, rated {rating}/10 by {int(row["vote_count"]):,} fans.'
        return {"tmdb_id": tid, "description": desc[:499]}
    # Last resort: highest scored movie not in history, with quality floor
    pool = _df[~_df["tmdb_id"].isin(history_ids)]
    pool = pool[pool["vote_average"] >= 7.0]
    pool = pool[pool["vote_count"] >= MIN_VOTE_COUNT]
    if banned_genres:
        pool = pool[~pool["genres"].apply(lambda g: any(bg in _safe(g) for bg in banned_genres))]
    if excluded_langs:
        pool = pool[~pool["original_language"].isin(excluded_langs)]
    if pool.empty:
        pool = _df[~_df["tmdb_id"].isin(history_ids)]
    row = pool.nlargest(1, "_score").iloc[0]
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "description": f'"{row["title"]}" ({_safe(row["year"])}) is one of the highest-rated films available - a critically acclaimed must-watch.'[:499],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list, history_ids: list = []) -> dict:
    """Return dict with 'tmdb_id' (int) and 'description' (str)."""
    history_id_set = {int(i) for i in history_ids}

    banned_genres = _detect_negative_genres(preferences)
    excluded_langs = _detect_excluded_languages(preferences)
    max_runtime = _detect_runtime_limit(preferences)
    banned_keywords = _detect_banned_keywords(preferences)

    candidates = _get_candidates(preferences, history_id_set, n=25, banned_keywords=banned_keywords)
    if candidates.empty:
        candidates = _get_candidates(preferences, set(), n=10, banned_keywords=banned_keywords)

    candidates = _boost_candidates(candidates, history_id_set)
    top10 = candidates.head(10)

    if top10.empty:
        return _fallback(_df, history_id_set, banned_genres, excluded_langs, max_runtime, banned_keywords)

    try:
        result = _two_stage_llm(preferences, history, top10, history_id_set)
        tid = int(result.get("tmdb_id", -1))
        desc = str(result.get("description", ""))
        if tid in VALID_IDS and tid not in history_id_set and desc:
            return {"tmdb_id": tid, "description": desc[:499]}
        raise ValueError("result failed validation")
    except Exception as e:
        print(f"[WARN] LLM failed ({type(e).__name__}: {e}), using fallback")
        return _fallback(candidates, history_id_set, banned_genres, excluded_langs, max_runtime, banned_keywords)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()

    preferences = (
        args.preferences.strip() if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip() if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start
    print(json.dumps(result, indent=2))
    print(f"\nServed in {elapsed:.2f}s")