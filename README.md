# Movie Recommendation Agent

A semantic movie recommendation agent. Given a user's free-text preferences and watch history, it returns a single `tmdb_id` and a short persuasive pitch from a database of 1,000 popular TMDB films.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# First run builds the ChromaDB index (~1 min, one-time only)
OLLAMA_API_KEY=your_key python llm.py --preferences "I love sci-fi thrillers"

# Run the test suite
OLLAMA_API_KEY=your_key python test.py

# Run the full evaluation harness
OLLAMA_API_KEY=your_key python evaluate.py
```

Get a free API key at [ollama.com/settings/keys](https://ollama.com/settings/keys).

---

## API Specification

### Endpoint
**POST** `/recommend`

### Request
```json
{
  "user_id": 120945,
  "preferences": "I love superheroes and feel-good buddy cop stories.",
  "history": [
    {"tmdb_id": 68721, "name": "Iron Man 3"},
    {"tmdb_id": 574475, "name": "Final Destination Bloodlines"}
  ]
}
```

### Response
```json
{
  "tmdb_id": 299536,
  "user_id": 120945,
  "description": "Infinity War raises the stakes of the entire marvel universe by pitting its greatest heroes against an impossible foe. The ensemble cast delivers career-best performances in this thrilling, emotional, and visually stunning masterpiece."
}
```

### Example cURL
```bash
curl -X POST https://your-deployment.leapcell.dev/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "preferences": "I love superheroes and feel-good buddy cop stories.",
    "history": [{"tmdb_id": 24428, "name": "The Avengers"}]
  }'
```

### Constraints
- Description: ≤500 characters
- Response time: <20 seconds
- Never recommends movies in `history`
- `tmdb_id` must exist in the 1,000-movie database

---

## Architecture

The pipeline has four stages that run on every request.

### Stage 1: Semantic Retrieval

All 1,000 movies are embedded into a local ChromaDB vector database using `sentence-transformers/all-MiniLM-L6-v2`, which runs entirely on the user's machine with no API calls. At inference time the user's query is embedded and the top 80 most semantically similar candidates are retrieved via cosine similarity.

Before embedding, the query goes through two enrichment steps:

**Query expansion** maps shorthand genre terms to richer search strings. For example, `"sci-fi"` expands to `"science fiction space future technology dystopia"`, and `"rom-com"` expands to `"romantic comedy love funny relationship dating"`.

**Non-literal translation** handles figurative requests by mapping them to concrete thematic terms before the vector search. Examples:

| User says | Translated to |
|---|---|
| "something so bad it's good" | "cult classic campy B-movie cheap cheesy absurd" |
| "a good cry" | "emotional tearjerker moving heartfelt" |
| "movie to watch with my grandma" | "gentle warm feel-good family comedy drama classic" |
| "turn my brain off" | "fun action spectacle easy entertaining" |
| "comfort movie" | "warm familiar feel-good comfort classic" |

**"Like X" thematic enrichment** handles requests referencing a specific film. When a query contains "like", "similar to", or "same as" alongside a known title, the actual themes of that film are appended. For example, "like Parasite but not Korean" becomes enriched with `"class struggle social satire dark thriller rich poor family infiltration"`.

### Stage 2: Filtering and Ranking

Hard filters are applied to the retrieved candidates using an O(1) pre-built lookup dictionary (no per-candidate DataFrame scans):

- **History exclusion**: any `tmdb_id` in the user's watch history is removed
- **Genre negation**: phrases like "no superhero movies", "tired of horror", or "superhero fatigue" detect banned genres and exclude them
- **Language exclusion**: phrases like "but not Korean" or "non-Japanese" map to ISO language codes and exclude matching films
- **Runtime filter**: "under 90 minutes", "short film", "quick watch" detect a max runtime and filter accordingly; films from 2025 or later are also excluded since their runtimes may be unverified
- **Content keyword filter**: catches thematic mismatches that genre tags miss. For example, "something peaceful" bans candidates whose `keywords` or `overview` contain "vampire", "zombie", "monster", "demon", etc. even if the genre tag says "Family"
- **Quality floor**: movies with fewer than 500 votes or a rating below 5.5 are excluded from the fallback path

After filtering, a soft boost re-ranks candidates that share a director or cast member with movies in the user's watch history, rewarding continuity.

### Stage 3: Two-Stage LLM Generation

The top 5 filtered candidates are passed to `gemma4:31b-cloud` via the Ollama API in two sequential calls:

**Stage 1 - Selection**: A compact prompt (title, year, genres, director only - no overviews) asks the model to pick the single best `tmdb_id` from the shortlist. The model is explicitly instructed to only use IDs from the provided list. If it returns an ID outside the candidates or in watch history, the top-ranked candidate is used instead.

**Stage 2 - Pitch writing**: The selected movie's full metadata (director, cast, tagline, overview) is passed to the model with instructions to write a persuasive 2-3 sentence pitch under 300 characters. The prompt detects the user's intended tone (upbeat, dark, intellectual) and instructs the model to match it.

Each stage has an independent timeout (9s for selection, 7s for pitch) with one retry on stage 1. This keeps the total worst-case time under 20 seconds.

### Stage 4: Deterministic Fallback

If either LLM call fails after retry, the system never crashes. A fallback function selects the highest-scored candidate from the filtered shortlist and constructs a readable description from the movie's metadata. The fallback respects all active filters (genre bans, language exclusions, runtime limits, content keyword bans) so it never recommends a clearly wrong film even without the LLM.

---

## Evaluation Strategy

The evaluation harness (`evaluate.py`) runs 14 fixed test cases and scores each recommendation on two axes using an LLM-as-a-judge.

### Test cases

8 baseline cases cover core genre scenarios (superhero action, hard sci-fi, family animated, avoid violence, etc.). 6 history-aware cases cover more nuanced situations:

- Superhero fatigue with Avengers history
- "More like Interstellar" with Interstellar excluded
- "Like Parasite but not Korean" with Parasite excluded
- Teenager into action/adventure with MCU history
- Grandma-friendly movie with history
- Nolan-style thriller with Nolan films excluded

### LLM-as-judge scoring

After each recommendation, the judge model scores the result 1-5 on two criteria:

- **Relevance**: does this movie actually match what the user asked for?
- **Persuasiveness**: would this pitch make someone want to watch it?

The judge's one-sentence comment explains the reasoning. This combination catches cases where the right movie gets a bad pitch (relevance 5, persuasiveness 2) and cases where an eloquent pitch is wasted on the wrong film (relevance 1, persuasiveness 5 — a failure mode we encountered early with "Django Unchained" for a violence-free request).

### Programmatic checks

Every test automatically fails if:
- The returned JSON is invalid
- `tmdb_id` is missing or not in the 1,000-movie candidate list
- The recommended movie is in the user's watch history
- The description exceeds 500 characters
- Total response time exceeds 20 seconds

### Build-Measure-Learn cycle

The evaluation harness was used iteratively throughout development. Each run produces `eval_results.json` with full scores, latencies, and judge comments. Key improvements driven by evaluation results:

- Discovered that "avoid violence" queries surfaced Hotel Transylvania (Comedy/Family genre tag hides "vampire" in keywords) → added content keyword filter
- Discovered that fallback descriptions scored 0-1 on persuasiveness ("Looking for action, comedy? Film X is rated 6.0") → rewrote fallback template
- Discovered that "like Parasite but not Korean" returned Mulan (no thematic match) → added MOVIE_THEME_MAP enrichment
- Discovered that runtime filter was ignored when LLM timed out → extended fallback to respect all active filters
- Tracked persuasiveness scores improving from 1.88 avg (all fallbacks, rate-limited session) to 4.33 avg (retry logic added) over development

### Evaluation Results

**14-Test Benchmark Results:**

| Metric | Score | Details |
|--------|-------|---------|
| **Programmatic Tests** | 14/14 ✅ | All tests: valid JSON, correct IDs, no watch history duplicates, descriptions ≤500 chars, responses <20s |
| **Relevance (LLM Judge)** | 3.5/5 avg | Range: 1–5. Strong on straightforward preferences (superhero action, rom-com, sci-fi). Weaker on niche requests (campy cult classics, runtime constraints). |
| **Persuasiveness (LLM Judge)** | 3.86/5 avg | Range: 2–5. Pitch quality consistently engaging. Fallback descriptions (3/5) less compelling than LLM-generated (4.5/5). |
| **Response Latency** | 5.14s avg | Range: 3.11–18.14s. One timeout on stage 1 (superhero action test, 18.14s) recovered via retry logic. |
| **Max Latency** | 18.14s | Still under 20s limit. Retry logic successfully handled timeout. |

**Notable Test Cases:**

| Test | Result | Judge Comment |
|------|--------|----------------|
| Superhero action | ✅ Pass | Relevance 5/5, Persuasiveness 4/5. Epic recommendation perfectly matches request. |
| Hard sci-fi | ✅ Pass | Relevance 5/5, Persuasiveness 4/5. Interstellar is a cerebral masterpiece match. |
| Superhero fatigue | ✅ Pass | Relevance 5/5, Persuasiveness 5/5. Perfect pivot to Oppenheimer; high-energy pitch. |
| Feel-good rom-com | ✅ Pass | Relevance 4/5, Persuasiveness 4/5. Beauty and the Beast fits all criteria. |
| Avoid violence ⚠️ | ❌ Fail | Relevance 1/5, Persuasiveness 4/5. Recommended "Scary Movie 5" (horror) when user asked for peaceful. Content keyword filter missed this edge case. |
| Runtime constraint | ❌ Fail | Relevance 1/5, Persuasiveness 3/5. "Downsizing" is 125 min, violates <90 min constraint. Runtime filter needs refinement. |
| Campy cult classic | ⚠️ Partial | Relevance 2/5, Persuasiveness 4/5. "The Substance" is high-quality satire, not campy B-movie vibe. Non-literal translation needs examples. |

**Key Findings:**

✅ **Strengths:**
- Semantic retrieval excels at genre matching and "like X" enrichment
- Two-stage LLM keeps recommendations in-distribution and descriptions compelling
- Retry logic prevents timeout failures in most cases
- Deterministic fallback ensures system robustness

⚠️ **Areas for Improvement:**
- Content keyword filter missed "Scary Movie 5" horror framing despite "avoid violence" request
- Runtime filter allowed 125-min film for "<90 min" request (filter may not be parsing correctly)
- Non-literal translation ("campy cult classic") benefits from more diverse examples

---

## Creativity & Novel Approaches

### 1. Semantic Retrieval with Query Enrichment

Instead of naive keyword matching, the system:
- **Query expansion**: Maps shorthand genre terms to rich semantic expansions (e.g., "rom-com" → "romantic comedy love funny relationship dating")
- **Non-literal translation**: Translates figurative language ("a good cry" → "emotional tearjerker moving heartfelt")
- **Movie thematic DNA**: Maps references like "like Parasite" to actual themes ("class struggle social satire dark thriller")

This allows ChromaDB to retrieve semantically relevant films even when users speak in idioms or reference other movies.

### 2. Multi-Stage LLM with Explicit Guardrails

Rather than one black-box LLM call, the system splits into two explicit stages:
- **Stage 1 (Selection)**: Explicitly constrain LLM to IDs in the shortlist; if it hallucinates, fall back deterministically
- **Stage 2 (Pitch)**: Write compelling descriptions with tone detection (upbeat vs. dark vs. intellectual)
- **Independent timeouts + retry**: Each stage has its own timeout with fallback recovery

This prevents the most common failure mode: returning a movie ID that doesn't exist or is in the watch history.

### 3. Content Keyword Filtering Beyond Genre Tags

Genre tags alone miss critical thematic content (e.g., "Family" includes Hotel Transylvania with vampires). The system adds a layer that scans `keywords` and `overview` text for banned content, catching edge cases where genre taxonomy fails.

### 4. Deterministic Fallback with Full Filter Respect

Rather than crashing or returning a generic error, if LLM calls timeout, the system:
- Selects the highest-ranked candidate from filtered results
- Constructs a readable description from metadata
- **Respects all active filters** (genre bans, language exclusions, runtime limits, content bans)

This ensures the system never violates user constraints even under adversarial conditions (timeouts, rate limits).

### 5. LLM-as-Judge Evaluation Framework

The evaluation harness uses a separate LLM instance to judge recommendations on two independent axes:
- **Relevance** (right movie for the request?)
- **Persuasiveness** (would this pitch sell it?)

This catches failure modes where relevance and persuasiveness diverge (e.g., "Django Unchained" is compelling but inappropriate for a "violence-free" request).

---

## Code Guide

```
llm.py                  Main implementation - all logic lives here
evaluate.py             14-test evaluation harness with LLM-as-judge
dspy_optimize.py        Offline DSPy prompt optimization script (run separately)
test.py                 Provided test suite (do not modify)
requirements.txt        Dependencies
tmdb_top1000_movies.csv Movie database (must be in same directory as llm.py)
.chroma_store/          ChromaDB index (auto-built on first run, gitignore this)
eval_results.json       Last evaluation output (auto-generated)
```

### Key constants in llm.py

| Constant | Value | Purpose |
|---|---|---|
| `MODEL` | `gemma4:31b-cloud` | Ollama model (do not change) |
| `LLM_TIMEOUT_SECONDS` | 9 | Stage 1 selection timeout |
| `STAGE2_TIMEOUT_SECONDS` | 7 | Stage 2 pitch timeout |
| `MIN_VOTE_COUNT` | 500 | Quality floor for candidates |
| `CHROMA_PATH` | `.chroma_store/` | Where the vector index lives |

### Rebuilding the ChromaDB index

If you change `MIN_VOTE_COUNT` or any metadata fields, delete `.chroma_store/` and rerun. The index name is versioned in code (`movies_v2`) — bumping it also forces a rebuild.

---

## Dependencies

```
ollama                  Ollama API client
pandas                  Data loading and manipulation
chromadb                Vector database for semantic retrieval
sentence-transformers   Local embedding model (all-MiniLM-L6-v2)
numpy                   Numerical operations
fastapi                 API server
uvicorn                 ASGI server
pydantic                Request/response validation
dspy-ai                 Offline prompt optimization
```

---

## Team Members

- Sofia Najera Gonzalez
- Jiale Guan
- Zhihan Zhang
- Doreen ZHU

---

## License

This project is part of the Spring 2026 Agentic AI assignment.
