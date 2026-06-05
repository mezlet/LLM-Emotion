#!/usr/bin/env python3
"""
scrape_rrlab_kb.py — RRLab website scraper for Self-RAG knowledge base
======================================================================
Crawls the entire https://rrlab.cs.rptu.de/en website, extracts clean text
from every page, chunks it into retrieval-sized passages, and indexes them
into ChromaDB for use by the Self-RAG module in vlm2.py.

Run this ONCE (or whenever you want to refresh the knowledge base):
    python scrape_rrlab_kb.py

Requirements:
    pip install chromadb sentence-transformers requests beautifulsoup4

Configuration:
    CHROMA_PERSIST_DIR  — where ChromaDB stores its index (must match self_rag.py)
    MAX_PAGES           — safety limit on number of pages crawled
    CHUNK_SIZE          — target word count per passage chunk
    CHUNK_OVERLAP       — word overlap between adjacent chunks (context continuity)
    DELAY_SECONDS       — polite delay between HTTP requests
"""

from __future__ import annotations

import re
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL            = "https://rrlab.cs.rptu.de/en"
CHROMA_PERSIST_DIR  = "chroma_db"          # must match self_rag.py
CHROMA_COLLECTION   = "emah_knowledge"     # must match self_rag.py
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"  # must match self_rag.py

MAX_PAGES           = 200    # crawl at most this many pages
CHUNK_SIZE          = 120    # target words per chunk
CHUNK_OVERLAP       = 20     # overlap words between chunks
DELAY_SECONDS       = 0.4    # seconds between requests (be polite)
REQUEST_TIMEOUT     = 15     # seconds

# Pages/path prefixes to skip (binary files, login pages, etc.)
SKIP_PATTERNS = [
    r"/fileadmin/",
    r"/typo3/",
    r"\.(pdf|jpg|jpeg|png|gif|svg|mp4|zip|docx|xlsx)$",
    r"tx_news_pi1",         # news detail pages with hash params
    r"cHash=",
    r"#",
]

# HTML elements whose text we always discard (nav, footer, etc.)
STRIP_TAGS = [
    "nav", "footer", "header", "script", "style",
    "noscript", "form", "button", "aside",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(url: str) -> bool:
    for pat in SKIP_PATTERNS:
        if re.search(pat, url, re.IGNORECASE):
            return True
    return False


def is_internal_en(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.netloc == "rrlab.cs.rptu.de"
        and parsed.path.startswith("/en")
    )


def normalise_url(url: str) -> str:
    """Strip fragment and query string, return canonical URL."""
    p = urlparse(url)
    return f"https://{p.netloc}{p.path}".rstrip("/")


def extract_text(soup: BeautifulSoup) -> str:
    """
    Extract main-content text from a parsed page.
    Strips navigation, footer, and boilerplate; preserves headings
    and paragraph structure as plain text.
    """
    # Remove noisy structural elements
    for tag in STRIP_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    # Try to isolate main content area
    main = (
        soup.find("main")
        or soup.find("div", id="main-content")
        or soup.find("div", class_=re.compile(r"content", re.I))
        or soup.find("body")
    )
    if main is None:
        return ""

    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = el.get_text(separator=" ", strip=True)
        if text and len(text) > 20:   # skip very short fragments
            lines.append(text)

    return "\n".join(lines)


def chunk_text(
    text: str,
    url: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """
    Split text into overlapping word-based chunks.
    Returns list of (chunk_text, source_url) tuples.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 60:   # skip trivially short chunks
            chunks.append((chunk, url))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── Crawler ───────────────────────────────────────────────────────────────────

def crawl_rrlab() -> list[tuple[str, str]]:
    """
    BFS crawl of rrlab.cs.rptu.de/en.
    Returns list of (chunk_text, source_url) for all discovered content.
    """
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; EMARRLabKBBot/1.0; "
        "+https://rrlab.cs.rptu.de/en/robots/ameca)"
    )

    visited:  set[str]       = set()
    queue:    deque[str]     = deque([BASE_URL])
    all_chunks: list[tuple[str, str]] = []

    print(f"Starting crawl of {BASE_URL}")
    print(f"Max pages: {MAX_PAGES}  |  Chunk size: {CHUNK_SIZE} words  |  Overlap: {CHUNK_OVERLAP}\n")

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        url = normalise_url(url)

        if url in visited or should_skip(url):
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"  [{resp.status_code}] SKIP  {url}")
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as exc:
            print(f"  [ERR]  {url} — {exc}")
            continue

        soup  = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title else url

        text   = extract_text(soup)
        chunks = chunk_text(text, url)
        all_chunks.extend(chunks)

        print(f"  [OK]  ({len(visited):>3}) {url}")
        print(f"         → {len(chunks)} chunks | title: {title[:60]}")

        # Discover new links
        for a in soup.find_all("a", href=True):
            full = normalise_url(urljoin(url, a["href"]))
            if (
                full not in visited
                and is_internal_en(full)
                and not should_skip(full)
            ):
                queue.append(full)

        time.sleep(DELAY_SECONDS)

    print(f"\nCrawl complete: {len(visited)} pages, {len(all_chunks)} total chunks")
    return all_chunks


# ── ChromaDB indexing ─────────────────────────────────────────────────────────

def build_knowledge_base(chunks: list[tuple[str, str]]) -> None:
    """
    Index all chunks into ChromaDB using SentenceTransformer embeddings.
    Wipes the existing collection and rebuilds from scratch.
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        print("\nERROR: chromadb / sentence-transformers not installed.")
        print("Run:  pip install chromadb sentence-transformers")
        sys.exit(1)

    print(f"\nBuilding ChromaDB index at '{CHROMA_PERSIST_DIR}'...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunks to index: {len(chunks)}\n")

    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete and recreate collection for a clean rebuild
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"Deleted existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    col = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert (ChromaDB handles embedding internally)
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch        = chunks[i : i + BATCH]
        texts        = [c[0] for c in batch]
        sources      = [c[1] for c in batch]
        ids          = [f"doc_{i + j}" for j in range(len(batch))]

        col.add(
            documents=texts,
            metadatas=[{"source": s} for s in sources],
            ids=ids,
        )

        pct = min(100, int((i + len(batch)) / len(chunks) * 100))
        print(f"  Indexed {i + len(batch):>5} / {len(chunks)} chunks  ({pct}%)")

    print(f"\nDone. Collection '{CHROMA_COLLECTION}' contains {col.count()} documents.")


# ── Verification query ────────────────────────────────────────────────────────

def verify_kb() -> None:
    """Run a quick sanity-check query against the freshly built index."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        return

    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col    = client.get_collection(CHROMA_COLLECTION, embedding_function=ef)

    test_queries = [
        "What is Ameca?",
        "Who leads the Robotics Research Lab?",
        "What research projects does RRLab have?",
        "Tell me about EMAH system",
    ]

    print("\n── Verification queries ─────────────────────────────────────────")
    for q in test_queries:
        results = col.query(query_texts=[q], n_results=1, include=["documents", "distances"])
        doc  = results["documents"][0][0]
        dist = results["distances"][0][0]
        print(f"\nQ: {q}")
        print(f"   dist={dist:.3f}  → {doc[:120]}...")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = crawl_rrlab()

    if not chunks:
        print("No content extracted. Check network access to rrlab.cs.rptu.de")
        sys.exit(1)

    build_knowledge_base(chunks)
    verify_kb()

    print("Knowledge base ready. Start vlm2.py to use Self-RAG.")