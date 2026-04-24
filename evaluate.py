"""
evaluate.py - Evaluation harness for the Movie Recommendation Agent

Usage:
    OLLAMA_API_KEY=your_key python evaluate.py
    OLLAMA_API_KEY=your_key python evaluate.py --no-judge   # skip LLM scoring
"""

import json
import os
import sys
import time
import statistics

import ollama
import pandas as pd

from llm import get_recommendation, _df as ALL_MOVIES

VALID_IDS = set(ALL_MOVIES["tmdb_id"].astype(int))
TIMEOUT_SECONDS = 20
JUDGE_MODEL = "gemma4:31b-cloud"

EVAL_TESTS = [
    # --- No history ---
    {
        "label": "superhero action",
        "preferences": "I love superhero action movies with epic battles.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "feel-good rom-com",
        "preferences": "I want something funny, light, and romantic. A feel-good movie.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "hard sci-fi",
        "preferences": "I love hard science fiction with complex ideas and space exploration.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "dark psychological thriller",
        "preferences": "I am in the mood for a dark, intense psychological thriller.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "family animated",
        "preferences": "I want something heartwarming and fun for the whole family. Animated is great.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "avoid violence",
        "preferences": "I want a great movie but without violence or horror. Something peaceful and thoughtful.",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "non-literal comfort movie",
        "preferences": "something so bad it is good, like a campy cult classic",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "runtime constraint",
        "preferences": "a short film night, under 90 minutes, nothing too heavy",
        "history": [],
        "history_ids": [],
    },
    # --- With history ---
    {
        "label": "superhero fatigue with history",
        "preferences": "I am tired of superhero movies. Give me something smart and dramatic.",
        "history": ["Avengers: Infinity War", "Avengers: Endgame"],
        "history_ids": [299536, 299534],
    },
    {
        "label": "more like Interstellar",
        "preferences": "I loved Interstellar, give me something with that same epic sci-fi feel.",
        "history": ["Interstellar"],
        "history_ids": [157336],
    },
    {
        "label": "like Parasite but not Korean",
        "preferences": "like Parasite but not Korean",
        "history": ["Parasite"],
        "history_ids": [496243],
    },
    {
        "label": "teenager with history",
        "preferences": "something my teenage son would love, he is into action and adventure",
        "history": ["Avengers: Endgame", "Spider-Man: No Way Home"],
        "history_ids": [299534, 634649],
    },
    {
        "label": "grandma movie with history",
        "preferences": "a movie to watch with my grandma, nothing violent or scary",
        "history": ["The Sound of Music"],
        "history_ids": [15121],
    },
    {
        "label": "avoid repeat director",
        "preferences": "I want a Christopher Nolan-style mind-bending thriller",
        "history": ["Inception", "Tenet", "Interstellar"],
        "history_ids": [27205, 577922, 157336],
    },
]


def _judge_score(preferences, title, description):
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    prompt = f"""You are a strict movie recommendation evaluator.

User asked for: "{preferences}"
Recommended: "{title}"
Pitch: "{description}"

Score from 1-5:
- relevance: how well does this match what the user asked for? (1=wrong, 5=perfect)
- persuasiveness: how compelling is the description? (1=boring/generic, 5=I must watch this NOW)

Reply with ONLY valid JSON, no markdown:
{{"relevance": <1-5>, "persuasiveness": <1-5>, "comment": "<one sentence>"}}"""
    try:
        r = client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        data = json.loads(r.message.content)
        return {
            "relevance": int(data.get("relevance", 0)),
            "persuasiveness": int(data.get("persuasiveness", 0)),
            "comment": str(data.get("comment", "")),
        }
    except Exception as e:
        return {"relevance": 0, "persuasiveness": 0, "comment": f"Judge error: {e}"}


def run_test(test, use_judge=True):
    label = test["label"]
    print(f"\n{'-'*60}")
    print(f"  TEST: {label}")
    print(f"  Preferences: {test['preferences'][:80]}")
    history_id_set = set(test["history_ids"])
    failures = []

    start = time.perf_counter()
    try:
        result = get_recommendation(test["preferences"], test["history"], test["history_ids"])
    except json.JSONDecodeError as e:
        failures.append(f"Invalid JSON: {e}")
        result = {}
    except Exception as e:
        failures.append(f"Exception: {e}")
        result = {}
    elapsed = time.perf_counter() - start

    # Programmatic checks
    if not isinstance(result, dict):
        failures.append("Return type is not dict")
    if "tmdb_id" not in result:
        failures.append("Missing 'tmdb_id' key")
    if "description" not in result:
        failures.append("Missing 'description' key")

    tmdb_id = None
    if "tmdb_id" in result:
        try:
            tmdb_id = int(result["tmdb_id"])
        except (TypeError, ValueError):
            failures.append(f"tmdb_id not castable to int: {result['tmdb_id']!r}")
        if tmdb_id is not None and tmdb_id not in VALID_IDS:
            failures.append(f"tmdb_id {tmdb_id} not in candidate list")
        if tmdb_id in history_id_set:
            failures.append(f"tmdb_id {tmdb_id} is already in watch history!")

    description = str(result.get("description", ""))
    if len(description) > 500:
        failures.append(f"Description too long: {len(description)} chars (max 500)")
    if elapsed > TIMEOUT_SECONDS:
        failures.append(f"Timeout: {elapsed:.1f}s > {TIMEOUT_SECONDS}s")

    title = "Unknown"
    if tmdb_id:
        row = ALL_MOVIES[ALL_MOVIES["tmdb_id"] == tmdb_id]
        if not row.empty:
            title = row.iloc[0]["title"]

    passed = len(failures) == 0
    print(f"  {'PASS' if passed else 'FAIL'} ({elapsed:.2f}s)")
    print(f"  Recommended: \"{title}\" (tmdb_id={tmdb_id})")
    print(f"  Description ({len(description)} chars): {description[:110]}{'...' if len(description) > 110 else ''}")
    for f in failures:
        print(f"  x {f}")

    judge = {"relevance": None, "persuasiveness": None, "comment": ""}
    if use_judge and passed and tmdb_id:
        print("  Judging...", end=" ", flush=True)
        judge = _judge_score(test["preferences"], title, description)
        print(f"Relevance={judge['relevance']}/5  Persuasiveness={judge['persuasiveness']}/5")
        print(f"  -> {judge['comment']}")

    return {
        "label": label,
        "passed": passed,
        "elapsed": elapsed,
        "tmdb_id": tmdb_id,
        "title": title,
        "description": description,
        "failures": failures,
        "relevance": judge["relevance"],
        "persuasiveness": judge["persuasiveness"],
        "comment": judge["comment"],
    }


def main():
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY is not set.")
        sys.exit(1)

    use_judge = "--no-judge" not in sys.argv
    print(f"\n{'='*60}")
    print(f"  MOVIE RECOMMENDER - EVALUATION HARNESS")
    print(f"  {len(EVAL_TESTS)} tests | LLM judge: {'ON' if use_judge else 'OFF'}")
    print(f"{'='*60}")

    results = []
    for i, t in enumerate(EVAL_TESTS):
        results.append(run_test(t, use_judge=use_judge))
        if i < len(EVAL_TESTS) - 1:
            time.sleep(1.5)  # brief pause between tests to avoid API rate limiting

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    rel_scores = [r["relevance"] for r in results if r["relevance"] is not None]
    pers_scores = [r["persuasiveness"] for r in results if r["persuasiveness"] is not None]
    latencies = [r["elapsed"] for r in results]

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Programmatic: {len(passed)}/{len(results)} passed")
    for r in failed:
        print(f"  x {r['label']}: {', '.join(r['failures'])}")

    if rel_scores:
        print(f"\n  LLM Judge (n={len(rel_scores)}):")
        print(f"    Relevance:       avg={statistics.mean(rel_scores):.2f}  min={min(rel_scores)}  max={max(rel_scores)}")
        print(f"    Persuasiveness:  avg={statistics.mean(pers_scores):.2f}  min={min(pers_scores)}  max={max(pers_scores)}")

    print(f"\n  Latency: avg={statistics.mean(latencies):.2f}s  max={max(latencies):.2f}s")

    out = "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out}")

    sys.exit(0 if len(passed) == len(results) else 1)


if __name__ == "__main__":
    main()