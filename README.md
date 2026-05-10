# SHL Conversational Assessment Recommender

A **FastAPI** service that acts as an expert SHL assessment consultant, helping
hiring managers find the right assessments through natural dialogue.

---

## Architecture

```
User ──► POST /chat ──► FastAPI (main.py)
                            │
                            ▼
                    agent.py (Gemini 2.5 Flash)
                            │
              ┌─────────────┴──────────────┐
              │                            │
        FAISS retrieval              Gemini API
        (catalog.py)                 (grounded generation)
              │
        all-MiniLM-L6-v2
        (local embeddings)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **FAISS in-memory** | Zero infra cost; 377 assessments fit trivially (< 5 MB). Rebuilt on cold start (< 10 s). |
| **Gemini 2.5 Flash** | Fast, free-tier available, large context window for catalog injection. |
| **all-MiniLM-L6-v2** | 80 MB, CPU-friendly, good semantic quality for short texts. |
| **Retrieval-augmented generation** | Top-60 catalog items injected per request so Gemini stays grounded and cannot hallucinate URLs. |
| **Stateless API** | Full conversation history sent on every call — no server-side session storage needed. |
| **Pydantic validation** | Schema enforcement on both input and output; deviations return clean 422 errors. |

---

## Project layout

```
shl-recommender/
├── main.py            # FastAPI app, /health and /chat endpoints
├── agent.py           # Gemini integration + response parsing
├── catalog.py         # Catalog loader, Assessment dataclass, CatalogIndex (FAISS)
├── test_agent.py      # Pytest suite (schema + behaviour probes + recall smoke tests)
├── requirements.txt
├── render.yaml        # Render.com deployment config
├── .env.example
└── data/
    └── product.json   # SHL catalog (377 assessments)
```

---

## Local setup

```bash
# 1. Clone & create virtualenv
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set env vars
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 4. Run
uvicorn main:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Chat example:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a mid-level Java developer who works with stakeholders"}
    ]
  }'
```

---

## Deploy to Render (free tier)

> **Cold start note**: `all-MiniLM-L6-v2` (~80 MB) is downloaded from HuggingFace on
> first boot. Render's free tier spins down after inactivity; the evaluator's 2-minute
> grace period on `/health` covers this.

---

## Running tests

```bash
# Requires GEMINI_API_KEY in environment
pytest test_agent.py -v
```

Test categories:
- **Schema tests** — response structure, URL validity, list bounds
- **Behaviour probes** — clarification, refusal, recommendation, refinement, comparison
- **Recall@K smoke tests** — grounded recommendations on 3 public-trace scenarios
- **Turn-limit compliance** — correct behaviour at conversation turn 8

---

## API reference

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request body:**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "...previous reply..."},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 3 assessments that fit ...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` while clarifying.
- `end_of_conversation` is `true` when the agent considers the task complete.
- `test_type` codes: `K`=Knowledge & Skills, `A`=Ability & Aptitude, `P`=Personality,
  `B`=Biodata/SJT, `C`=Competencies, `D`=Development/360, `E`=Exercises.

---

## Evaluation approach

### Recall@K
Every recommended URL is verified against the scraped catalog. The agent is
prompted with an explicit constraint: "Every URL must come verbatim from the
catalog data below." Retrieval (FAISS top-60) ensures only real catalog items
are presented to the model for selection.

### Behaviour probes tested
| Probe | How it's handled |
|---|---|
| Vague query | Agent asks clarifying questions, `recommendations: []` |
| Off-topic request | Agent politely refuses, stays in scope |
| Prompt injection | Ignored; system prompt cannot be overridden by user content |
| Refinement | History passed in full; agent updates shortlist incrementally |
| Comparison | Agent answers using catalog descriptions only |
| Hallucination | FAISS retrieval + explicit system prompt rule prevent invention |

### What didn't work (and what we changed)
- **Full catalog in prompt** — 377 items × description text exceeded practical token
  limits and increased latency. Solution: FAISS retrieval to top 60.
- **Asking model to output plain text + JSON** — model mixed formats. Solution:
  strict system prompt requiring JSON-only output; robust regex extraction as fallback.
- **Gemini returning markdown fences** — occasional. Solution: regex strip in
  `_parse_agent_response`.
