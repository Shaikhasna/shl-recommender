# SHL Assessment Recommender

A conversational agent that takes hiring managers from vague intent to a grounded shortlist of SHL Individual Test Solutions — through dialogue, not keyword search.

**Stack:** FastAPI · Groq (llama-3.3-70b-versatile) · In-memory catalog search · Railway

---

## Why no vector DB?

The SHL Individual Test Solutions catalog is ~200-300 items — small enough to fit relevant subsets directly into the LLM context window. Adding ChromaDB or FAISS would introduce cold-start latency, extra dependencies, and failure points with no accuracy gain at this scale. Keyword search over catalog fields (name, description, test_type, languages) retrieves the top 30 candidates per query in <5ms, which are then passed as grounded context to the LLM.

---

## Architecture

```
POST /chat
    │
    ├── catalog.py: keyword search → top 30 relevant assessments
    │       (tokenize query, score by field overlap, rank)
    │
    ├── agent.py: build prompt = system + catalog subset + conversation history
    │       → Groq API (llama-3.3-70b-versatile, temp=0.1)
    │       → parse JSON response
    │       → validate every URL against catalog set (O(1) lookup)
    │       → strip hallucinated items silently
    │
    └── main.py: Pydantic schema validation → typed ChatResponse
```

**Stateless design:** Every `/chat` call carries the full conversation history. No session storage, no DB writes per request. This matches the evaluator's replay harness exactly.

---

## Agent behavior

| Situation | Agent action |
|---|---|
| Vague query ("I need an assessment") | Asks ONE clarifying question, `recommendations: []` |
| Enough context (role + seniority + purpose) | Returns 1–10 assessments from catalog |
| Mid-conversation refinement | Updates shortlist, explains what changed |
| Comparison question | Answers from catalog data only, no hallucination |
| Off-topic / legal / prompt injection | Refuses, `recommendations: []` |
| User confirms shortlist | `end_of_conversation: true` |

Turn cap: 8 turns max (evaluator constraint). Each call responds in <5s well within the 30s timeout.

---

## Project structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI — /health + /chat, Pydantic validation
│   ├── agent.py         # Groq LLM calls, prompt construction, URL validation
│   └── catalog.py       # Catalog loader (strict=False JSON), keyword search
├── data/
│   └── catalog.json     # SHL Individual Test Solutions (downloaded once)
├── eval.py              # Recall@10 + schema compliance + behavior probes
├── download_catalog.py  # One-time catalog fetch utility
├── railway.toml         # Railway deployment config
└── requirements.txt
```

---

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a mid-level Java developer, 4 years experience, selection"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Also add a personality test"}
  ]
}
```

**Response:**
```json
{
  "reply": "Updated shortlist — added OPQ32r for personality screening.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules (non-negotiable):**
- `recommendations` → `[]` when clarifying or refusing
- `recommendations` → 1–10 items when committing to shortlist
- `end_of_conversation` → `true` only on user confirmation
- Every URL validated against scraped catalog before response is sent

---

## Local setup

```bash
git clone https://github.com/your-username/shl-recommender
cd shl-recommender
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Add your Groq key (free at console.groq.com)
echo "GROQ_API_KEY=gsk_..." > .env

# Download catalog
python download_catalog.py   # or manually save catalog.json to data/

# Run
uvicorn app.main:app --reload
# → http://localhost:8000/health
```

## Run evaluation

Copy `sample_conversations.zip` into `data/`, then:
```bash
python eval.py
```
Outputs: schema compliance rate, Mean Recall@10, behavior probe pass-rate.

---

## Deployment (Railway)

```bash
git push origin main
# Railway auto-detects railway.toml
# Set env var: GROQ_API_KEY=gsk_...
# Public URL: https://your-app.up.railway.app
```

Health check path: `/health` — Railway pings this on deploy.