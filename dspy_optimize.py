"""
dspy_optimize.py — Offline DSPy prompt optimization for the Movie Recommender

This script runs OFFLINE (never during live API serving). It uses DSPy's
BootstrapFewShot optimizer and an LLM-as-judge metric to automatically find
the best instruction wording for the two-stage prompts. Once finished, it
prints the optimized prompt strings which you paste into llm.py as the
SELECTION_PROMPT_TEMPLATE and PITCH_PROMPT_TEMPLATE constants.

Usage:
    OLLAMA_API_KEY=your_key python dspy_optimize.py

Requirements:
    pip install dspy-ai
"""

import json
import os
import sys
import statistics

import dspy
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configure DSPy to use Ollama
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

def _configure_dspy():
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("ERROR: OLLAMA_API_KEY is not set.")
        sys.exit(1)
    lm = dspy.LM(
        f"openai/{MODEL}",
        api_base="https://ollama.com/v1",
        api_key=api_key,
        max_tokens=512,
    )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
_df = pd.read_csv(DATA_PATH)
_df["tmdb_id"] = _df["tmdb_id"].astype(int)
_df["_score"] = _df["vote_average"] * (
    _df["vote_count"] / (_df["vote_count"] + 500)
) + 6.5 * (500 / (_df["vote_count"] + 500))
VALID_IDS = set(_df["tmdb_id"].tolist())


def _safe(v, limit=0):
    s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
    return s[:limit] if limit else s


# ---------------------------------------------------------------------------
# Training examples
# ---------------------------------------------------------------------------

TRAIN_EXAMPLES = [
    {
        "preferences": "I love superhero action movies with epic battles.",
        "history": [],
        "ideal_genres": ["Action", "Science Fiction", "Adventure"],
    },
    {
        "preferences": "Something funny and romantic, a feel-good date night movie.",
        "history": [],
        "ideal_genres": ["Comedy", "Romance"],
    },
    {
        "preferences": "I want a dark psychological thriller that messes with my mind.",
        "history": [],
        "ideal_genres": ["Thriller", "Drama", "Crime"],
    },
    {
        "preferences": "Hard science fiction about space exploration and big ideas.",
        "history": [],
        "ideal_genres": ["Science Fiction", "Adventure", "Drama"],
    },
    {
        "preferences": "A powerful drama with incredible acting and emotional depth.",
        "history": [],
        "ideal_genres": ["Drama"],
    },
    {
        "preferences": "Heartwarming animated movie the whole family can enjoy.",
        "history": [],
        "ideal_genres": ["Animation", "Family", "Comedy"],
    },
    {
        "preferences": "I'm tired of superhero movies. Give me something original and thoughtful.",
        "history": ["Avengers: Endgame"],
        "ideal_genres": ["Drama", "Crime", "Thriller"],
    },
    {
        "preferences": "A gripping war movie with realistic battle scenes.",
        "history": [],
        "ideal_genres": ["War", "Drama", "Action"],
    },
]


def _build_candidate_snippet(preferences, history_ids=None):
    """Return a small shortlist of relevant candidates for a training example."""
    if history_ids is None:
        history_ids = set()
    top = _df[~_df["tmdb_id"].isin(history_ids)].nlargest(10, "_score")
    lines = []
    for r in top.itertuples():
        lines.append(
            f'- tmdb_id={int(r.tmdb_id)} | "{r.title}" ({_safe(r.year)}) '
            f'| genres: {_safe(r.genres)} '
            f'| director: {_safe(r.director) or "N/A"} '
            f'| cast: {_safe(r.top_cast)[:60]} '
            f'| overview: {_safe(r.overview)[:150]}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

class SelectBestMovie(dspy.Signature):
    """Given a user's movie preferences and a shortlist of candidates,
    select the single best matching movie tmdb_id."""

    preferences: str = dspy.InputField(desc="What the user wants to watch")
    history_text: str = dspy.InputField(desc="Movies the user has already seen (do not recommend)")
    movie_list: str = dspy.InputField(desc="Candidate movies with metadata")
    tmdb_id: int = dspy.OutputField(desc="The tmdb_id of the best matching movie")
    reason: str = dspy.OutputField(desc="Brief 10-word rationale for the selection")


class WritePersuasivePitch(dspy.Signature):
    """Write a persuasive movie recommendation pitch tailored to the user's preferences.
    The pitch must be under 480 characters, name a director or actor, and convey
    the emotional experience rather than summarising the plot."""

    preferences: str = dspy.InputField(desc="What the user wants to watch")
    title: str = dspy.InputField(desc="Movie title")
    year: str = dspy.InputField(desc="Release year")
    director: str = dspy.InputField(desc="Director name")
    cast: str = dspy.InputField(desc="Lead actor name")
    tagline: str = dspy.InputField(desc="Movie tagline")
    overview: str = dspy.InputField(desc="Short movie overview")
    mood_hint: str = dspy.InputField(desc="Tone guidance for the pitch")
    description: str = dspy.OutputField(desc="Persuasive pitch under 480 characters")


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------

class MovieSelector(dspy.Module):
    def __init__(self):
        self.select = dspy.ChainOfThought(SelectBestMovie)

    def forward(self, preferences, history_text, movie_list):
        return self.select(
            preferences=preferences,
            history_text=history_text,
            movie_list=movie_list,
        )


class PitchWriter(dspy.Module):
    def __init__(self):
        self.write = dspy.ChainOfThought(WritePersuasivePitch)

    def forward(self, preferences, title, year, director, cast, tagline, overview, mood_hint):
        return self.write(
            preferences=preferences,
            title=title,
            year=year,
            director=director,
            cast=cast,
            tagline=tagline,
            overview=overview,
            mood_hint=mood_hint,
        )


# ---------------------------------------------------------------------------
# Metric functions (LLM-as-judge)
# ---------------------------------------------------------------------------

class JudgeRelevance(dspy.Signature):
    """Judge whether the selected movie is a good match for the user's preferences."""
    preferences: str = dspy.InputField()
    movie_genres: str = dspy.InputField()
    ideal_genres: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Relevance score 0.0-1.0")


class JudgePersuasiveness(dspy.Signature):
    """Judge whether the pitch description would convince someone to watch the movie."""
    preferences: str = dspy.InputField()
    description: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Persuasiveness score 0.0-1.0")


_judge = dspy.ChainOfThought(JudgeRelevance)
_persuasion_judge = dspy.ChainOfThought(JudgePersuasiveness)


def selection_metric(example, prediction, trace=None):
    """Score the selection module: genre match + validity checks."""
    try:
        tmdb_id = int(prediction.tmdb_id)
    except (TypeError, ValueError):
        return 0.0

    if tmdb_id not in VALID_IDS:
        return 0.0

    row = _df[_df["tmdb_id"] == tmdb_id]
    if row.empty:
        return 0.0
    genres = _safe(row.iloc[0].get("genres"))

    ideal = example.ideal_genres
    overlap = sum(1 for g in ideal if g in genres)
    genre_score = overlap / max(len(ideal), 1)

    try:
        judge_result = _judge(
            preferences=example.preferences,
            movie_genres=genres,
            ideal_genres=", ".join(ideal),
        )
        llm_score = float(judge_result.score)
    except Exception:
        llm_score = genre_score

    return 0.4 * genre_score + 0.6 * llm_score


def pitch_metric(example, prediction, trace=None):
    """Score the pitch module: length constraint + persuasiveness."""
    description = str(prediction.description)
    if len(description) > 499:
        return 0.0
    if len(description) < 50:
        return 0.0

    try:
        result = _persuasion_judge(
            preferences=example.preferences,
            description=description,
        )
        score = float(result.score)
    except Exception:
        score = 0.5

    return score


# ---------------------------------------------------------------------------
# Build DSPy training sets
# ---------------------------------------------------------------------------

def _build_selection_trainset():
    examples = []
    for ex in TRAIN_EXAMPLES:
        history_ids = set()
        snippet = _build_candidate_snippet(ex["preferences"], history_ids)
        history_text = ", ".join(f'"{h}"' for h in ex["history"]) if ex["history"] else "none"
        examples.append(
            dspy.Example(
                preferences=ex["preferences"],
                history_text=history_text,
                movie_list=snippet,
                ideal_genres=ex["ideal_genres"],
            ).with_inputs("preferences", "history_text", "movie_list")
        )
    return examples


def _build_pitch_trainset():
    examples = []
    for ex in TRAIN_EXAMPLES:
        # Pick a plausible movie for this preference
        top = _df.nlargest(5, "_score").iloc[0]
        examples.append(
            dspy.Example(
                preferences=ex["preferences"],
                title=_safe(top.get("title")),
                year=_safe(top.get("year")),
                director=_safe(top.get("director")) or "unknown",
                cast=_safe(top.get("top_cast")).split(",")[0].strip() or "ensemble",
                tagline=_safe(top.get("tagline")) or "none",
                overview=_safe(top.get("overview"))[:300],
                mood_hint="Match the emotional tone of what the user is asking for.",
            ).with_inputs("preferences", "title", "year", "director", "cast",
                          "tagline", "overview", "mood_hint")
        )
    return examples


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize_selector():
    print("\n" + "="*60)
    print("Optimizing SELECTION prompt with BootstrapFewShot...")
    print("="*60)

    trainset = _build_selection_trainset()
    module = MovieSelector()

    optimizer = dspy.BootstrapFewShot(
        metric=selection_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized


def optimize_pitcher():
    print("\n" + "="*60)
    print("Optimizing PITCH prompt with BootstrapFewShot...")
    print("="*60)

    trainset = _build_pitch_trainset()
    module = PitchWriter()

    optimizer = dspy.BootstrapFewShot(
        metric=pitch_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _configure_dspy()

    print("Running DSPy optimization — this calls the LLM many times.")
    print("Expected runtime: 5-15 minutes depending on API speed.\n")

    # Optimize selector
    try:
        opt_selector = optimize_selector()
        selector_history = opt_selector.select.dump_state()
        print("\n[Selector] Optimized instructions:")
        print(json.dumps(selector_history, indent=2))
    except Exception as e:
        print(f"[WARN] Selector optimization failed: {e}")
        selector_history = None

    # Optimize pitcher
    try:
        opt_pitcher = optimize_pitcher()
        pitcher_history = opt_pitcher.write.dump_state()
        print("\n[Pitcher] Optimized instructions:")
        print(json.dumps(pitcher_history, indent=2))
    except Exception as e:
        print(f"[WARN] Pitcher optimization failed: {e}")
        pitcher_history = None

    # Save compiled states
    out = {
        "selector": selector_history,
        "pitcher": pitcher_history,
    }
    out_path = os.path.join(os.path.dirname(__file__), "dspy_optimized.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[Done] Optimized prompts saved to: {out_path}")
    print("Review the output above and paste any improved instructions into")
    print("SELECTION_PROMPT_TEMPLATE and PITCH_PROMPT_TEMPLATE in llm.py.")


if __name__ == "__main__":
    main()
