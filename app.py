import os
from typing import Iterable, List

import streamlit as st
from openai import OpenAI
from surrealdb import Surreal


# ----------------------------
# Config helpers
# ----------------------------

def get_secret(key: str, default: str | None = None) -> str | None:
    # Streamlit secrets first (Cloud), then env (local)
    if key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SURREAL_URL = get_secret("SURREALDB_URL")          # e.g. wss://<INSTANCE_ENDPOINT>
SURREAL_NS = get_secret("SURREALDB_NS", "chat")
SURREAL_DB = get_secret("SURREALDB_DB", "chat")
SURREAL_USER = get_secret("SURREALDB_USER")
SURREAL_PASS = get_secret("SURREALDB_PW")

CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = get_secret("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# text-embedding-3-small defaults to 1536 dims, -3-large defaults to 3072 dims
# (you *must* match SurrealDB index dimension to this).
EMBED_DIMS = int(get_secret("OPENAI_EMBED_DIMS", "1536"))

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY (set it in Streamlit Secrets).")
    st.stop()
if not (SURREAL_URL and SURREAL_USER and SURREAL_PASS):
    st.error("Missing SurrealDB connection secrets (SURREALDB_URL, SURREALDB_USER, SURREALDB_PW).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# DB connection (cached)
# ----------------------------



@st.cache_resource
def get_db():
    url = get_secret("SURREALDB_URL")
    ns = get_secret("SURREALDB_NS", "chat")
    dbname = get_secret("SURREALDB_DB", "chat")
    user = get_secret("SURREALDB_USER")
    pw = get_secret("SURREALDB_PW")
    token = get_secret("SURREALDB_TOKEN")  # optional (leave unset if not using)

    db = Surreal(url)

    # Some SDK builds require connect(); others don't have it.
    if hasattr(db, "connect"):
        db.connect()

    if token:
        # Token auth path
        db.authenticate(token)
    else:
        # Username/password path
        db.signin({
            "namespace": ns,
            "database": dbname,
            "username": user,
            "password": pw,
        })
        db.use(ns, dbname)
        db.query("RETURN 1;")

    db.use(ns, dbname)

    # Fail fast: proves endpoint + auth + ns/db are valid
    db.query("RETURN 1;")
    return db

def ensure_schema(db: Surreal) -> None:
    # HNSW is the modern vector index option in SurrealDB
    # (dimension must match embedding length). See SurrealDB vector index docs.
    schema = f"""
    DEFINE TABLE IF NOT EXISTS chunks;

    DEFINE FIELD IF NOT EXISTS text ON chunks TYPE string;
    DEFINE FIELD IF NOT EXISTS source ON chunks TYPE string;

    DEFINE FIELD IF NOT EXISTS embedding ON chunks TYPE array<float>;

    DEFINE INDEX IF NOT EXISTS chunks_embedding_hnsw
        ON chunks FIELDS embedding
        HNSW DIMENSION {EMBED_DIMS} DIST COSINE;
    """
    db.query(schema)

# ----------------------------
# RAG utilities
# ----------------------------

def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        i = max(j - overlap, i + 1)
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Batch embeddings request
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [row.embedding for row in resp.data]

def store_chunks(db: Surreal, chunks: List[str], source: str) -> None:
    vectors = embed_texts(chunks)
    rows = [{"text": t, "source": source, "embedding": v} for t, v in zip(chunks, vectors)]
    # Create multiple records in one call (SDK supports list payloads)
    db.create("chunks", rows)

def retrieve_context(db: Surreal, query: str, k: int = 5, ef: int = 120) -> List[str]:
    q_vec = embed_texts([query])[0]

    # SurrealDB KNN operator supports (k, ef) form.
    # We keep k/ef as server-side constants (safe ints), and bind only the vector.
    sql = f"""
    LET $q := $vec;
    SELECT text, source, vector::distance::knn() AS distance
    FROM chunks
    WHERE embedding <|{k},{ef}|> $q
    ORDER BY distance
    LIMIT {k};
    """
    res = db.query(sql, {"vec": q_vec})

    # SDK returns a list of statement results; each has a "result" list
    try:
        rows = (res or [])[0].get("result", [])
    except Exception:
        rows = []

    return [r.get("text", "") for r in rows if isinstance(r, dict) and r.get("text")]

def stream_answer(messages: List[dict]) -> Iterable[str]:
    # Streaming via Responses API
    # We yield text deltas for Streamlit's st.write_stream
    with client.responses.stream(model=CHAT_MODEL, input=messages) as stream:
        for event in stream:
            # openai-python example checks event.type contains "output_text"
            if hasattr(event, "type") and "output_text" in event.type and hasattr(event, "delta"):
                if event.delta:
                    yield event.delta

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Rugby Strength Coach (RAG)", layout="wide")
st.title("Rugby Strength Coach (SurrealDB + OpenAI)")

try:
    db = get_db()
except Exception as e:
    st.error(f"SurrealDB connection failed: {e!r}")
    st.stop()
ensure_schema(db)

with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
    if uploaded:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        source = uploaded.name
        if st.button("Ingest into SurrealDB"):
            chunks = split_text(raw)
            store_chunks(db, chunks, source=source)
            st.success(f"Ingested {len(chunks)} chunks from {source}")

    st.divider()
    st.header("Chat settings")
    top_k = st.slider("Top-K context chunks", 1, 10, 5)
    st.caption("Tip: start by uploading your rugby programming notes / routines.")

st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about training (strength, power, speed, conditioning, recovery)...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_chunks = retrieve_context(db, prompt, k=top_k)
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    system = (
        "You are a rugby strength & conditioning coach. "
        "Prioritize safety, progressive overload, and rugby-specific outputs (power, speed, repeatability). "
        "Use the provided context when relevant; if context is missing, say what you assume."
    )

    # Responses API supports role-based message arrays as input
    messages = [{"role": "system", "content": system}]
    if context_text:
        messages.append({"role": "system", "content": f"CONTEXT:\n{context_text}"})
    messages.extend({"role": x["role"], "content": x["content"]} for x in st.session_state.messages)

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(messages))

    st.session_state.messages.append({"role": "assistant", "content": answer})