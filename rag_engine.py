"""
rag_engine.py
=============
Core Retrieval-Augmented Generation (RAG) engine for the FAQ Chatbot.

Responsibilities
----------------
- **Ingestion**: Load a plain-text FAQ file, split it into chunks, and embed
  each chunk via the OpenAI Embeddings API.
- **Indexing**: Store the resulting vectors in a FAISS flat L2 index and
  persist both the index and the raw chunks to disk so they survive restarts.
- **Retrieval**: Given a user query, embed it and search the index for the
  top-K nearest neighbours, returning the matching text chunks.
- **Demo mode**: Provide a lightweight keyword-overlap search against a JSON
  FAQ file so the app works without an OpenAI API key.

Public API
----------
    build_index(faq_path)          -> (faiss.Index, list[str])
    load_index()                   -> (faiss.Index, list[str])
    retrieve(query, index, chunks) -> list[str]
    search_faqs_json(query)        -> (str, str) | None
    embed_texts(client, texts)     -> np.ndarray
    load_chunks(filepath)          -> list[str]
    get_client()                   -> OpenAI

Constants
---------
    INDEX_PATH   : Path where the FAISS binary index is saved.
    CHUNKS_PATH  : Path where the pickled chunk list is saved.
    EMBED_MODEL  : OpenAI embedding model name.
    TOP_K        : Default number of chunks to retrieve per query.
"""

import os
import json
import pickle
import re
import numpy as np
import faiss
from openai import OpenAI

# ── File paths and model constants ────────────────────────────────────────────
FAQ_JSON_PATH = "faqs.json"   # Demo-mode structured FAQ data
INDEX_PATH    = "faiss.index" # Persisted FAISS index (binary)
CHUNKS_PATH   = "chunks.pkl"  # Persisted chunk list (pickle)
EMBED_MODEL   = "text-embedding-3-small"  # 1536-dim, cheapest OpenAI embedding
TOP_K         = 4             # Number of chunks returned per query


# ── Demo-mode helpers ─────────────────────────────────────────────────────────

def load_faqs_json(path: str = FAQ_JSON_PATH) -> list[dict]:
    """Load the structured FAQ JSON file used in demo mode.

    The file is expected to be a JSON array where each element is an object
    with at least ``"question"`` and ``"answer"`` string fields.

    Args:
        path: Filesystem path to the JSON file. Defaults to ``FAQ_JSON_PATH``.

    Returns:
        A list of FAQ dicts, e.g. ``[{"question": "...", "answer": "..."}, ...]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> set[str]:
    """Convert *text* to a set of lowercase alphanumeric tokens.

    Punctuation is stripped so that "FAQ?" and "FAQ" map to the same token,
    improving keyword-overlap matching in demo mode.

    Args:
        text: Any string to tokenize.

    Returns:
        A set of lowercase word strings with no punctuation.
    """
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def search_faqs_json(query: str, path: str = FAQ_JSON_PATH) -> tuple[str, str] | None:
    """Find the best matching FAQ entry using keyword overlap (demo mode).

    Scoring heuristic
    -----------------
    For each FAQ entry the score is::

        score = 2 × |query_tokens ∩ question_tokens|
                  + |query_tokens ∩ answer_tokens|

    Question matches are weighted 2× because a question that shares words
    with the user's query is a stronger signal than answer text alone.

    Args:
        query: The user's natural-language question.
        path:  Path to the JSON FAQ file. Defaults to ``FAQ_JSON_PATH``.

    Returns:
        A ``(question, answer)`` tuple for the best-scoring entry, or ``None``
        if no entry scored above zero (i.e. no shared tokens were found).
    """
    faqs = load_faqs_json(path)
    query_words = _tokenize(query)
    best_score = 0
    best_match = None

    for faq in faqs:
        question_words = _tokenize(faq["question"])
        answer_words   = _tokenize(faq["answer"])
        score = (
            len(query_words & question_words) * 2  # question overlap weighted 2×
            + len(query_words & answer_words)
        )
        if score > best_score:
            best_score = score
            best_match = faq

    if best_match and best_score > 0:
        return best_match["question"], best_match["answer"]
    return None


# ── OpenAI client ─────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """Create and return an authenticated OpenAI client.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable (which
    ``python-dotenv`` populates from ``.env`` on startup).

    Returns:
        An initialised :class:`openai.OpenAI` instance ready for API calls.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set or is empty.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def load_chunks(filepath: str) -> list[str]:
    """Read a plain-text FAQ file and split it into paragraph chunks.

    Splitting strategy
    ------------------
    The file is split on blank lines (``\\n\\n``). Within each block, lines
    beginning with ``#`` (after stripping whitespace) are treated as comments
    and removed. Empty blocks are discarded.

    This means a typical ``faqs.txt`` looks like::

        # Section heading (ignored)
        Q: What is your return policy?
        A: We offer 30-day returns on all items.

        Q: How do I contact support?
        A: Email us at support@example.com.

    Each double-newline-separated block becomes one chunk.

    Args:
        filepath: Path to the plain-text FAQ file (e.g. ``"faqs.txt"``).

    Returns:
        A list of non-empty text strings, one per FAQ paragraph.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    for block in raw.split("\n\n"):
        # Strip comment lines (lines whose first non-whitespace char is '#')
        lines = [line for line in block.splitlines()
                 if not line.strip().startswith("#")]
        chunk = "\n".join(lines).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of strings using the OpenAI Embeddings API.

    Sends all texts in a single batched API request for efficiency. The
    returned vectors are cast to ``float32`` because FAISS requires that
    dtype (it does not accept ``float64``).

    Args:
        client: An authenticated :class:`openai.OpenAI` instance.
        texts:  A non-empty list of strings to embed. Each string is one
                document or query.

    Returns:
        A NumPy array of shape ``(len(texts), 1536)`` and dtype ``float32``,
        where row *i* is the embedding for ``texts[i]``.

    Raises:
        openai.AuthenticationError: If the API key is invalid.
        openai.RateLimitError:      If the rate limit is exceeded.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


# ── Index management ──────────────────────────────────────────────────────────

def build_index(faq_path: str = "faqs.txt") -> tuple[faiss.Index, list[str]]:
    """Chunk a FAQ file, embed all chunks, build a FAISS index, and save to disk.

    Pipeline
    --------
    1. ``load_chunks(faq_path)``            — parse file into text blocks
    2. ``embed_texts(client, chunks)``      — call OpenAI embeddings API
    3. ``faiss.IndexFlatL2(dim).add(vecs)`` — build exact L2 index in memory
    4. Write index to ``INDEX_PATH``        — binary FAISS format
    5. Pickle chunks to ``CHUNKS_PATH``     — parallel list for result lookup

    The index type (``IndexFlatL2``) performs exact nearest-neighbour search.
    It is appropriate for FAQ sets up to ~100 k chunks where the precision of
    exact search outweighs the speed benefit of approximate methods.

    Args:
        faq_path: Path to the plain-text FAQ source file. Defaults to
                  ``"faqs.txt"``.

    Returns:
        A ``(index, chunks)`` tuple where *index* is the populated
        :class:`faiss.IndexFlatL2` and *chunks* is the list of raw text
        strings whose positions correspond to index vectors.

    Raises:
        ValueError:       If *faq_path* contains no parseable chunks.
        FileNotFoundError: If *faq_path* does not exist.
        openai.OpenAIError: On any API failure during embedding.
    """
    client = get_client()
    chunks = load_chunks(faq_path)
    if not chunks:
        raise ValueError(f"No content found in {faq_path}")

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_texts(client, chunks)

    # Build index — dim is inferred from the embedding model output (1536)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Persist to disk so the index survives server restarts
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built: {len(chunks)} chunks, dim={dim}")
    return index, chunks


def load_index() -> tuple[faiss.Index, list[str]]:
    """Load a previously built FAISS index and its chunk list from disk.

    Both ``INDEX_PATH`` (FAISS binary) and ``CHUNKS_PATH`` (pickle) must
    exist; they are written together by :func:`build_index`.

    Returns:
        A ``(index, chunks)`` tuple ready for use with :func:`retrieve`.

    Raises:
        FileNotFoundError: If either ``INDEX_PATH`` or ``CHUNKS_PATH`` is
                           missing — typically meaning :func:`build_index`
                           has never been run for this directory.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Index not found. Run build_index() or use the sidebar to ingest your FAQs."
        )
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    top_k: int = TOP_K,
) -> list[str]:
    """Embed a query and return the *top_k* most relevant FAQ chunks.

    The query is embedded with the same model used during indexing so the
    vectors are comparable. FAISS returns indices into *chunks*; any index
    of ``-1`` indicates FAISS could not fill the requested *top_k* slots
    (only possible when the index has fewer vectors than *top_k*) and is
    silently skipped.

    Args:
        query:  The user's natural-language question.
        index:  A populated :class:`faiss.Index` built by :func:`build_index`.
        chunks: The parallel list of raw text strings matching the index
                vectors — i.e. ``chunks[i]`` is the text for vector ``i``.
        top_k:  Maximum number of chunks to return. Defaults to ``TOP_K`` (4).

    Returns:
        A list of up to *top_k* text strings, ordered by ascending L2
        distance (most relevant first).

    Raises:
        ValueError:        If ``OPENAI_API_KEY`` is not set.
        openai.OpenAIError: On any API failure during query embedding.
    """
    client = get_client()

    # Embed the query into the same vector space as the indexed chunks
    q_vec = embed_texts(client, [query])  # shape: (1, 1536)

    # FAISS returns (distances, indices) arrays of shape (n_queries, top_k)
    _distances, indices = index.search(q_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1:  # -1 means FAISS ran out of neighbours
            results.append(chunks[idx])
    return results
