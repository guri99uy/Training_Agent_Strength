# Rugby Strength Coach (Streamlit + SurrealDB + OpenAI)

A minimal Streamlit app that:
1) ingests training notes into SurrealDB as vectorized chunks  
2) retrieves the most relevant chunks for a question (RAG)  
3) answers like a rugby strength & conditioning coach using OpenAI

## What you get
- **Streamlit UI**: upload `.txt` / `.md` → ingest → chat
- **SurrealDB**: persistent knowledge base (`chunks` table + vector index)
- **OpenAI**: embeddings + chat responses

---

## Repository structure

```
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ CONTRIBUTING.md
```

---

## Requirements
- Python 3.10+
- A running SurrealDB instance (recommended: SurrealDB Cloud)
- An OpenAI API key

---

## Run locally

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Set environment variables
Create a `.env` locally (optional) or export env vars:

```bash
export OPENAI_API_KEY="..."
export SURREALDB_URL="wss://<INSTANCE_ENDPOINT>"
export SURREALDB_NS="chat"
export SURREALDB_DB="chat"
export SURREALDB_USER="..."
export SURREALDB_PW="..."

# optional
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"
export OPENAI_EMBED_DIMS="1536"
```

### 3) Start the app
```bash
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud

1) Push this repo to GitHub  
2) Create a new Streamlit app pointing to the repo  
3) In Streamlit → **App settings → Secrets**, add:

```toml
OPENAI_API_KEY = "..."
SURREALDB_URL = "wss://<INSTANCE_ENDPOINT>"
SURREALDB_NS = "chat"
SURREALDB_DB = "chat"
SURREALDB_USER = "..."
SURREALDB_PW = "..."

# optional
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIMS = "1536"
```

---

## Notes on embeddings / dimensions
The vector index dimension **must match** the embedding model output size:
- `text-embedding-3-small` → typically **1536**
- `text-embedding-3-large` → typically **3072**

If you change the embedding model, update `OPENAI_EMBED_DIMS` accordingly.

---

## Data model (current)
- `chunks`: stores `{ text, source, embedding }`

Planned next:
- `profile`: player profile (position, goals, equipment, constraints)
- `workout_log`: session logs for personalized progression

---

## Security
- Do **not** commit API keys.
- Use Streamlit Secrets on Streamlit Cloud.

---

## Roadmap
- Add user profile + workout logging tables
- Add “plan generator” mode (weekly block programming)
- Add progress dashboard (volume, intensity, readiness)
