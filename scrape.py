#!/usr/bin/env python3
"""
scrape2.py — RRLab website scraper for Ameca Self-RAG
=====================================================

This script crawls the English RRLab website:

    https://rrlab.cs.rptu.de/en

It extracts readable text from reachable internal English pages, removes common
TYPO3/navigation noise, chunks the content into retrieval-friendly passages,
deduplicates it, adds category/priority metadata, and indexes everything into
ChromaDB so your Ameca/VLM chatbot can use it for grounded Self-RAG answers.

Run:
    python scrape2.py

Install:
    pip install chromadb sentence-transformers requests beautifulsoup4 lxml

If lxml is not available, the script automatically falls back to html.parser.

Important:
    The ChromaDB path and collection below must match your main Ameca script.

Main improvements in this version:
    - stronger TYPO3 content-block isolation
    - removes repeated navigation/header/footer phrases from chunks
    - smaller, more specific chunks for MiniLM retrieval
    - category metadata: staff, project, robot, publication, teaching, conference
    - priority metadata for authoritative pages
    - category-aware verification
    - hybrid reranking during verification
    - source-level deduplication after reranking
    - robot-specific smaller chunks
    - teaching-page penalties for staff/project retrieval
    - low-value URL/content filtering before indexing
    - stronger archive/former-staff/publication-stub exclusion
    - /en/seite homepage canonicalization
    - category caps for balanced retrieval
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, FeatureNotFound


# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "https://rrlab.cs.rptu.de/en"

# Must match your main Self-RAG script.
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION = "emah_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_PAGES = 320
REQUEST_TIMEOUT = 15
DELAY_SECONDS = 0.35

# Smaller chunks improve precision with MiniLM-style embedding models.
# This is especially useful for staff, project, and robot pages.
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

MIN_TEXT_CHARS = 80
MIN_CHUNK_CHARS = 80

VERIFY_TOP_K = 12
VERIFY_KEEP_TOP_N = 5
VERIFY_MAX_DISTANCE = 0.58

# Hard caps prevent one category from dominating the vector DB.
# This improves retrieval balance and keeps the KB semantically dense.
CATEGORY_LIMITS = {
    "robot": 220,
    "staff": 140,
    "conference": 15,
    "teaching": 35,
}

USER_AGENT = (
    "Mozilla/5.0 (compatible; AmecaSelfRAGBot/1.2; "
    "+https://rrlab.cs.rptu.de/en)"
)

SITEMAP_CANDIDATES = [
    "https://rrlab.cs.rptu.de/sitemap.xml",
    "https://rrlab.cs.rptu.de/en/sitemap.xml",
]

SKIP_PATTERNS = [
    r"/fileadmin/",
    r"/typo3/",
    r"/typo3conf/",
    r"/uploads/",
    r"/_assets/",
    r"/media/",
    r"/print",
    r"/login",
    r"/search",
    r"/datenschutzerklaerung",
    r"/impressum",
    r"\.(pdf|jpg|jpeg|png|gif|svg|webp|mp4|mov|avi|zip|rar|7z|doc|docx|xls|xlsx|ppt|pptx)$",
    r"tx_news_pi1",
    r"cHash=",
]

# These pages are reachable, but they usually add low-value semantic noise:
# - old conference schedules
# - repeated WFAR/GI pages
# - archived teaching pages from old semesters
# - tiny individual publication stub pages
#
# They can pollute retrieval because they repeat robotics/staff terms without
# being useful for answering normal RRLab questions.
LOW_VALUE_PATTERNS = [
    # Conference noise
    r"/conferences/.*/program",
    r"/conferences/.*/committee",
    r"/conferences/.*/venue",
    r"/conferences/.*/topics",
    r"/conferences/.*/schedule",
    r"/conferences/wfar",
    r"/conferences/gi-informatik",
    r"/conferences/ams07",

    # Old teaching archives with one-year suffixes, e.g. -17, -18, -19, -20, -21
    r"/teaching/offered-courses/.*-17$",
    r"/teaching/offered-courses/.*-18$",
    r"/teaching/offered-courses/.*-19$",
    r"/teaching/offered-courses/.*-20$",
    r"/teaching/offered-courses/.*-21$",

    # Archived semester labels and old course folders
    r"/teaching/offered-courses/.*/ws",
    r"/teaching/offered-courses/.*/ss",

    # Former staff pages are low-value for normal current RRLab answers.
    r"/staff/former-staff-members/",

    # Tiny publication stubs. Keep only publication overview pages such as
    # textbooks, dissertations, proceedings, media coverage, and complete list.
    r"/research/publications/[a-z0-9\-]+$",
]

HARD_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
]

# Selectors that usually contain real TYPO3/RRLab page content.
# These are tried before falling back to broad body extraction.
CONTENT_BLOCK_SELECTORS = [
    "div.ce-bodytext",
    "div.frame-type-text",
    "div.frame-type-textmedia",
    "div.frame-type-html",
    "div.frame",
    "article",
    "main",
]

TYPO3_NOISE_PATTERNS = [
    r"RPTU Department of Computer Science",
    r"Robotics Research Lab Home",
    r"Home Group Overview",
    r"Group Overview Group Overview",
    r"Head of the Laboratory Secretariat Technical Staff",
    r"Secretariat Technical Staff Research Associates",
    r"Research Associates Student Research Assistants Former Staff Members",
    r"Research Research Areas Control Architectures Outdoor Robots Indoor-Robots Humanoid Robots Simulation",
    r"Projects Finished Projects Publications Textbooks Dissertations RRLab Proceedings Media Coverage Complete List of Publications",
    r"Teaching Offered Courses Archive Examination appointments Student Research Work Templates Student Management",
    r"Instagram Facebook YouTube LinkedIn",
    r"Department of Computer Science in Kaiserslautern",
    r"Paul-Ehrlich-Straße building 46",
    r"Page \d+ of \d+",
    r"next previous",
    r"previous next",
    r"Read more More about",
    r"More about .*? More about",
]

BOILERPLATE_EXACT = {
    "skip navigation",
    "main navigation",
    "breadcrumb",
    "privacy policy",
    "data privacy",
    "imprint",
    "contact",
    "search",
    "menu",
    "instagram",
    "facebook",
    "youtube",
    "linkedin",
    "next",
    "previous",
}

BOILERPLATE_CONTAINS = [
    "javascript must be enabled",
    "to top",
    "print this page",
    "page 1 of",
]


@dataclass(frozen=True)
class PageRecord:
    url: str
    title: str
    text: str
    category: str
    priority: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source: str
    title: str
    category: str
    priority: int
    chunk_index: int


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    print(message, flush=True)


# =============================================================================
# BeautifulSoup helpers
# =============================================================================

def make_soup(markup: str, parser: str = "lxml") -> BeautifulSoup:
    try:
        return BeautifulSoup(markup, parser)
    except FeatureNotFound:
        return BeautifulSoup(markup, "html.parser")


# =============================================================================
# URL helpers
# =============================================================================

def should_skip(url: str) -> bool:
    if any(re.search(pattern, url, re.IGNORECASE) for pattern in SKIP_PATTERNS):
        return True

    if any(re.search(pattern, url, re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS):
        return True

    return False


def normalize_url(url: str) -> str:
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    scheme = "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path)

    # TYPO3 duplicate homepage alias. This prevents indexing both /en and /en/seite.
    if path == "/en/seite":
        path = "/en"

    if path != "/en":
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", "", ""))


def is_internal_english_page(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "rrlab.cs.rptu.de" and parsed.path.startswith("/en")


def detect_category(url: str) -> str:
    path = urlparse(url).path.lower()

    if "/staff/" in path or "/group-overview/head-of-the-laboratory" in path:
        return "staff"

    if path.endswith("/research/projects") or "/research/projects/" in path:
        return "project"

    if path.endswith("/robots") or "/robots/" in path:
        return "robot"

    if "/research/publications/" in path:
        return "publication"

    if path.endswith("/research/data-sets") or "/research/data-sets/" in path:
        return "dataset"

    if path.endswith("/research/research-areas") or "/research/research-areas/" in path or path.endswith("/research"):
        return "research_area"

    if "/teaching/" in path:
        return "teaching"

    if "/conferences/" in path:
        return "conference"

    if "/group-overview" in path:
        return "group"

    return "general"


def detect_priority(url: str, title: str) -> int:
    path = urlparse(url).path.lower()
    title_l = title.lower()

    if path.endswith("/en") or path.endswith("/en/seite"):
        return 90

    if "/group-overview/head-of-the-laboratory" in path:
        return 100

    if "/staff/research-associates/" in path:
        return 85

    if path.endswith("/staff/research-associates"):
        return 80

    if path.endswith("/research/projects"):
        return 95

    if "/research/projects/" in path and "/finished-projects/" not in path:
        return 90

    if "/research/projects/finished-projects" in path:
        return 50

    if path.endswith("/robots"):
        return 95

    if "/robots/ameca" in path:
        return 100

    if "/robots/" in path:
        return 85

    if "/research/research-areas/" in path:
        return 80

    if "/research/publications/complete-list-of-publications" in path:
        return 70

    if "/conferences/" in path:
        return 30

    if "/former-staff-members/" in path:
        return 35

    if "archive" in title_l:
        return 30

    return 50


def is_low_value_page(url: str, title: str, text: str, category: str) -> bool:
    """
    Content-level filtering after extraction.

    URL filters catch most low-value pages, but some pages pass through because
    URLs are not always predictable. This function removes pages that are not
    useful enough for conversational Self-RAG grounding.
    """
    path = urlparse(url).path.lower()
    title_l = title.lower().strip()
    text_l = text.lower()

    # Keep these high-value overview pages even when short.
    if path.endswith("/en") or path.endswith("/en/seite"):
        return False

    if "head-of-the-laboratory" in path:
        return False

    if "/robots/ameca" in path:
        return False

    if path.endswith("/research/projects"):
        return False

    if path.endswith("/robots"):
        return False

    if path.endswith("/staff/research-associates"):
        return False

    # Drop very short publication stubs. They usually contain only citation data
    # and hurt retrieval more than they help.
    if category == "publication" and len(text) < 900:
        return True

    # Drop conference pages that are mostly schedules/committees.
    if category == "conference":
        if len(text) < 1200:
            return True
        if any(token in title_l for token in ["schedule", "organizing committee", "programm"]):
            return True

    # Drop old archived course pages unless they are high-level teaching overview pages.
    if category == "teaching":
        keep_teaching = [
            "/teaching",
            "/teaching/offered-courses",
            "/teaching/examination-appointments",
            "/teaching/student-research-work",
            "/teaching/templates",
            "/teaching/student-management",
        ]
        if path not in keep_teaching:
            return True

    # Drop pages that are mostly navigation leftovers.
    navigation_tokens = [
        "home group overview",
        "secretariat technical staff",
        "research associates student research assistants",
        "instagram facebook youtube linkedin",
    ]
    if sum(1 for token in navigation_tokens if token in text_l) >= 2 and len(text) < 1500:
        return True

    return False


# =============================================================================
# HTTP and extraction
# =============================================================================

def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en,en-US;q=0.9",
        }
    )
    return session


def fetch_html(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
            and content_type.strip() != ""
        ):
            return None

        return response.text

    except requests.RequestException as exc:
        log(f"  [ERR] {url} — {exc}")
        return None


def clean_whitespace(text: str) -> str:
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_typo3_noise(text: str) -> str:
    text = clean_whitespace(text)

    for pattern in TYPO3_NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove repeated generic menu fragments that often appear after broad extraction.
    text = re.sub(
        r"Content\s+RPTU\s+Department\s+of\s+Computer\s+Science\s+Robotics\s+Research\s+Lab\s+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\b(next|previous)\b\s+\b(next|previous)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_title(soup: BeautifulSoup, fallback_url: str) -> str:
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return clean_whitespace(title)

    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        title = re.sub(r"\s*\|\s*.*$", "", title).strip()
        if title:
            return clean_whitespace(title)

    return fallback_url


def remove_hard_noise(soup: BeautifulSoup) -> None:
    for tag_name in HARD_STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def choose_best_content_container(soup: BeautifulSoup):
    selectors = [
        "main",
        "article",
        "section",
        "div#main-content",
        "div.content",
        "div.main-content",
        "div.tx-news",
        "div.frame",
        "div.ce-bodytext",
        "div.container",
        "body",
    ]

    candidates = []

    for selector in selectors:
        for candidate in soup.select(selector):
            text = remove_typo3_noise(candidate.get_text(" ", strip=True))
            text_len = len(text)
            if text_len > 0:
                candidates.append((text_len, candidate))

    if not candidates:
        return soup

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def remove_obvious_boilerplate_lines(lines: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for line in lines:
        line = remove_typo3_noise(line)
        line = clean_whitespace(line)
        if not line:
            continue

        lowered = line.lower().strip()

        if lowered in BOILERPLATE_EXACT:
            continue

        if any(token in lowered for token in BOILERPLATE_CONTAINS):
            continue

        if len(line) < 3:
            continue

        key = re.sub(r"\s+", " ", lowered)
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(line)

    return cleaned


def get_text_from_content_blocks(soup: BeautifulSoup) -> str:
    """
    Prefer real TYPO3 content blocks before broad body/container extraction.
    This reduces navigation noise such as:
        "Content RPTU Department of Computer Science Robotics Research Lab Home..."
    """
    block_texts: list[str] = []
    seen = set()

    for selector in CONTENT_BLOCK_SELECTORS:
        for block in soup.select(selector):
            lines: list[str] = []

            for element in block.find_all(
                ["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "figcaption"],
                recursive=True,
            ):
                line = element.get_text(" ", strip=True)
                line = clean_whitespace(line)
                if line:
                    lines.append(line)

            if not lines:
                raw = block.get_text("\n", strip=True)
                lines = [line.strip() for line in raw.splitlines() if line.strip()]

            lines = remove_obvious_boilerplate_lines(lines)
            text = remove_typo3_noise("\n".join(lines))

            if len(text) < 120:
                continue

            key = re.sub(r"\s+", " ", text.lower()).strip()
            if key in seen:
                continue

            seen.add(key)
            block_texts.append(text)

    return clean_whitespace("\n\n".join(block_texts))


def extract_clean_text(html: str, url: str) -> PageRecord | None:
    soup = make_soup(html, "lxml")
    remove_hard_noise(soup)

    title = extract_title(soup, url)
    category = detect_category(url)
    priority = detect_priority(url, title)

    # First: use TYPO3 content blocks.
    text = get_text_from_content_blocks(soup)

    # Second: fallback to best broad content container.
    if len(text) < MIN_TEXT_CHARS:
        container = choose_best_content_container(soup)

        structured_lines: list[str] = []
        for element in container.find_all(
            ["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "figcaption"],
            recursive=True,
        ):
            line = element.get_text(" ", strip=True)
            line = clean_whitespace(line)
            if line:
                structured_lines.append(line)

        structured_lines = remove_obvious_boilerplate_lines(structured_lines)

        if structured_lines:
            text = "\n".join(structured_lines)
        else:
            raw_text = container.get_text("\n", strip=True)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            lines = remove_obvious_boilerplate_lines(lines)
            text = "\n".join(lines)

        text = remove_typo3_noise(text)

    # Third: final fallback to body text.
    if len(text) < MIN_TEXT_CHARS:
        body = soup.find("body") or soup
        raw_text = body.get_text("\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        lines = remove_obvious_boilerplate_lines(lines)
        text = remove_typo3_noise("\n".join(lines))

    text = clean_whitespace(text)

    if len(text) < MIN_TEXT_CHARS:
        return None

    if is_low_value_page(url=url, title=title, text=text, category=category):
        return None

    return PageRecord(
        url=url,
        title=title,
        text=text,
        category=category,
        priority=priority,
    )


def discover_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        absolute = normalize_url(urljoin(current_url, anchor["href"]))

        if not is_internal_english_page(absolute):
            continue

        if should_skip(absolute):
            continue

        links.append(absolute)

    return links


# =============================================================================
# Sitemap discovery
# =============================================================================

def discover_sitemap_urls(session: requests.Session) -> list[str]:
    urls: list[str] = []

    for sitemap_url in SITEMAP_CANDIDATES:
        try:
            response = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                continue

            soup = make_soup(response.text, "xml")
            locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

            for loc in locs:
                normalized = normalize_url(loc)

                if is_internal_english_page(normalized) and not should_skip(normalized):
                    urls.append(normalized)

        except Exception:
            continue

    seen = set()
    deduped = []

    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)

    return deduped


# =============================================================================
# Crawling
# =============================================================================

def crawl_rrlab() -> list[PageRecord]:
    session = create_session()

    initial_urls = [normalize_url(BASE_URL)]
    sitemap_urls = discover_sitemap_urls(session)

    if sitemap_urls:
        log(f"Discovered {len(sitemap_urls)} English URLs from sitemap.")
        initial_urls.extend(sitemap_urls)

    queue: deque[str] = deque(initial_urls)
    queued: set[str] = set(initial_urls)
    visited: set[str] = set()
    pages: list[PageRecord] = []

    log(f"Starting crawl of {BASE_URL}")
    log(f"Max pages: {MAX_PAGES} | Delay: {DELAY_SECONDS}s\n")

    while queue and len(visited) < MAX_PAGES:
        url = normalize_url(queue.popleft())

        if url in visited:
            continue

        if should_skip(url):
            continue

        if not is_internal_english_page(url):
            continue

        visited.add(url)

        html = fetch_html(session, url)

        if not html:
            continue

        soup = make_soup(html, "lxml")
        page = extract_clean_text(html, url)

        if page:
            pages.append(page)
            log(f"  [OK]   ({len(visited):>3}) {url}")
            log(
                f"         title='{page.title[:80]}' | "
                f"category={page.category} | priority={page.priority} | chars={len(page.text)}"
            )
        else:
            log(f"  [SKIP] ({len(visited):>3}) {url} | low text content")

        for link in discover_links(soup, url):
            if link not in visited and link not in queued:
                queue.append(link)
                queued.add(link)

        time.sleep(DELAY_SECONDS)

    log(f"\nCrawl complete: visited={len(visited)}, usable_pages={len(pages)}")
    return pages


# =============================================================================
# Chunking and deduplication
# =============================================================================

def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_for_dedup(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9äöüß\s:/.,;()&+-]", "", text)
    return text.strip()


def chunk_page(page: PageRecord) -> list[ChunkRecord]:
    words = page.text.split()

    if not words:
        return []

    # Robot pages are often long and dense. Smaller chunks make retrieval more
    # precise for questions such as "What is Ameca?", "Tell me about RAVON",
    # or "What robots does RRLab have?"
    if page.category == "robot":
        local_chunk_size = 120
        local_chunk_overlap = 30
    else:
        local_chunk_size = CHUNK_SIZE
        local_chunk_overlap = CHUNK_OVERLAP

    chunks: list[ChunkRecord] = []
    start = 0
    chunk_index = 0
    step = max(1, local_chunk_size - local_chunk_overlap)

    while start < len(words):
        end = min(start + local_chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        chunk_text = remove_typo3_noise(chunk_text)

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            titled_chunk = (
                f"Title: {page.title}\n"
                f"Category: {page.category}\n"
                f"Source: {page.url}\n\n"
                f"{chunk_text}"
            ).strip()

            chunks.append(
                ChunkRecord(
                    chunk_id=f"{stable_hash(page.url)}_{chunk_index}",
                    text=titled_chunk,
                    source=page.url,
                    title=page.title,
                    category=page.category,
                    priority=page.priority,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

        if end >= len(words):
            break

        start += step

    return chunks


def apply_category_limits(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """
    Cap noisy/large categories before indexing.

    The goal is not to index the maximum amount of text. The goal is to index
    the most useful and balanced knowledge for retrieval.
    """
    grouped: dict[str, list[ChunkRecord]] = {}

    for chunk in chunks:
        grouped.setdefault(chunk.category, []).append(chunk)

    limited: list[ChunkRecord] = []

    for category, items in grouped.items():
        limit = CATEGORY_LIMITS.get(category)

        # Sort so high-priority authoritative pages survive category caps.
        items = sorted(
            items,
            key=lambda item: (
                -int(item.priority),
                item.source,
                item.chunk_index,
            ),
        )

        if limit is not None:
            items = items[:limit]

        limited.extend(items)

    # Stable final order for reproducible IDs/logs.
    limited.sort(key=lambda item: (item.category, -int(item.priority), item.source, item.chunk_index))
    return limited


def build_chunks(pages: list[PageRecord]) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    seen: set[str] = set()

    for page in pages:
        for chunk in chunk_page(page):
            dedup_key = normalize_for_dedup(chunk.text)

            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            all_chunks.append(chunk)

    before_limits = len(all_chunks)
    all_chunks = apply_category_limits(all_chunks)

    log(f"Prepared {before_limits} unique chunks from {len(pages)} pages.")
    log(f"After category caps: {len(all_chunks)} chunks remain.")

    category_counts: dict[str, int] = {}
    for chunk in all_chunks:
        category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1

    log("Chunk categories:")
    for category, count in sorted(category_counts.items()):
        cap = CATEGORY_LIMITS.get(category)
        if cap is None:
            log(f"  - {category}: {count}")
        else:
            log(f"  - {category}: {count} / cap {cap}")

    return all_chunks


# =============================================================================
# ChromaDB indexing
# =============================================================================

def build_knowledge_base(chunks: list[ChunkRecord]) -> None:
    if not chunks:
        log("No chunks to index.")
        sys.exit(1)

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        log("\nERROR: chromadb / sentence-transformers not installed.")
        log("Run: pip install chromadb sentence-transformers")
        sys.exit(1)

    log(f"\nBuilding ChromaDB index at '{CHROMA_PERSIST_DIR}'...")
    log(f"Collection: {CHROMA_COLLECTION}")
    log(f"Embedding model: {EMBEDDING_MODEL}")
    log(f"Chunks to index: {len(chunks)}\n")

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        client.delete_collection(CHROMA_COLLECTION)
        log(f"Deleted existing collection '{CHROMA_COLLECTION}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        collection.add(
            documents=[item.text for item in batch],
            metadatas=[
                {
                    "source": item.source,
                    "title": item.title,
                    "category": item.category,
                    "priority": item.priority,
                    "chunk_index": item.chunk_index,
                    "kind": "rrlab_website",
                }
                for item in batch
            ],
            ids=[item.chunk_id for item in batch],
        )

        indexed = start + len(batch)
        pct = int(indexed / len(chunks) * 100)
        log(f"  Indexed {indexed:>5} / {len(chunks)} chunks ({pct}%)")

    log(f"\nDone. Collection '{CHROMA_COLLECTION}' contains {collection.count()} documents.")


# =============================================================================
# Verification and hybrid reranking
# =============================================================================

def infer_query_category(query: str) -> str | None:
    q = query.lower()

    if any(token in q for token in ["ashita", "professor", "head", "leader", "leads", "staff", "research associate", "who is"]):
        return "staff"

    if any(token in q for token in ["project", "current project", "finished project", "sembai", "senna", "casrew", "zukunftbau"]):
        return "project"

    if any(token in q for token in ["robot", "robots", "ameca", "ravon", "robin", "unimog", "carl"]):
        return "robot"

    if any(token in q for token in ["publication", "paper", "textbook", "dissertation"]):
        return "publication"

    if any(token in q for token in ["course", "teaching", "seminar", "exam"]):
        return "teaching"

    if any(token in q for token in ["research area", "researches", "research"]):
        return "research_area"

    return None


def rewrite_query_for_verification(query: str) -> str:
    q = query.lower().strip()

    if "who leads" in q or "head" in q and "laboratory" in q:
        return "head of laboratory professor robotics research lab"

    if "current projects" in q:
        return "current RRLab projects SEmbAI SENNA CASREW ZukunftBau project"

    if "what is ameca" in q:
        return "Ameca Emah humanoid robot RRLab student companion"

    if "what robots" in q:
        return "RRLab robots Ameca RAVON Robin Unimog CARL robot platforms"

    if "ashita" in q:
        return "M. Sc. Ashita Ashok Ameca Robothespian human robot interaction trust expectation alignment"

    return query


def query_collection(collection, query: str, category: str | None = None, top_k: int = VERIFY_TOP_K):
    kwargs = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"],
    }

    if category:
        kwargs["where"] = {"category": category}

    return collection.query(**kwargs)


def hybrid_score(distance: float, metadata: dict, inferred_category: str | None, query: str, doc: str) -> float:
    """
    Higher is better.

    Chroma cosine distance: lower is better.
    Convert to similarity with 1 - distance, then add small bonuses.
    """
    similarity = 1.0 - float(distance)
    score = similarity

    category = metadata.get("category")
    priority = int(metadata.get("priority", 0) or 0)
    title = str(metadata.get("title", "")).lower()
    source = str(metadata.get("source", "")).lower()
    doc_l = doc.lower()
    q = query.lower()

    if inferred_category and category == inferred_category:
        score += 0.12

    if priority >= 100:
        score += 0.12
    elif priority >= 90:
        score += 0.08
    elif priority >= 80:
        score += 0.04

    if "/former-staff-members/" in source:
        score -= 0.08

    if category == "conference":
        score -= 0.10

    # Prevent category leakage. This matters for questions like "What is Ameca?"
    # where staff pages may mention Ameca but should not outrank the robot page.
    if inferred_category == "robot" and category == "staff":
        score -= 0.18

    if inferred_category == "robot" and category == "conference":
        score -= 0.25

    if inferred_category == "project" and category not in {"project", "general"}:
        score -= 0.12

    if inferred_category == "staff" and category not in {"staff", "general"}:
        score -= 0.12

    # Prevent teaching pages from leaking into staff/project retrieval.
    # This happened because course pages can mention staff names and robotics terms.
    if inferred_category == "project" and category == "teaching":
        score -= 0.15

    if inferred_category == "staff" and category == "teaching":
        score -= 0.10

    # Former staff pages are useful historically, but should not outrank current
    # staff pages for "who is..." or "who leads..." questions.
    if inferred_category == "staff" and "/former-staff-members/" in source:
        score -= 0.06

    if "head of laboratory" in q or "who leads" in q:
        if "head of the laboratory" in title or "head-of-the-laboratory" in source:
            score += 0.25

    if "ashita" in q:
        if "ashita ashok" in title or "ashita ashok" in doc_l:
            score += 0.25

    if "ameca" in q or "emah" in q:
        if "/robots/ameca" in source or "title: emah" in doc_l:
            score += 0.25

    if "project" in q:
        if "/research/projects/" in source and "/finished-projects/" not in source:
            score += 0.10
        if "/finished-projects/" in source and "current" in q:
            score -= 0.18

    # Prefer general overview pages when asking broad questions about RRLab.
    if ("what does rrlab research" in q or "research area" in q) and category == "research_area":
        score += 0.14

    return score


def flatten_results(results: dict) -> list[dict]:
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    rows = []
    seen_sources_chunks = set()

    for idx, doc in enumerate(docs):
        distance = float(distances[idx]) if idx < len(distances) else math.inf
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        key = (metadata.get("source", ""), metadata.get("chunk_index", idx))

        if key in seen_sources_chunks:
            continue
        seen_sources_chunks.add(key)

        rows.append(
            {
                "doc": doc,
                "distance": distance,
                "metadata": metadata,
            }
        )

    return rows


def print_hybrid_results(label: str, rows: list[dict], query: str, inferred_category: str | None) -> None:
    log(f"\n{label}")

    if not rows:
        log("   No result.")
        return

    scored = []
    for row in rows:
        score = hybrid_score(
            distance=row["distance"],
            metadata=row["metadata"],
            inferred_category=inferred_category,
            query=query,
            doc=row["doc"],
        )
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    # Strict source-level deduplication:
    # show only one best chunk per page so a single page does not dominate the
    # final context or verification output.
    displayed_pages = set()
    displayed = 0

    for score, row in scored:
        metadata = row["metadata"]
        source = metadata.get("source", "")

        if source in displayed_pages:
            continue

        displayed_pages.add(source)
        displayed += 1

        distance = row["distance"]
        category = metadata.get("category", "unknown")
        priority = metadata.get("priority", 0)
        title = metadata.get("title", "unknown")
        snippet = re.sub(r"\s+", " ", row["doc"][:280]).strip()
        status = "KEEP" if distance <= VERIFY_MAX_DISTANCE else "WEAK"

        log(
            f"   #{displayed}: hybrid={score:.3f} | distance={distance:.3f} [{status}] | "
            f"category={category} | priority={priority} | {title} | {source}"
        )
        log(f"       {snippet}...")

        if displayed >= VERIFY_KEEP_TOP_N:
            break

def verify_kb() -> None:
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        return

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(
            CHROMA_COLLECTION,
            embedding_function=embedding_function,
        )
    except Exception as exc:
        log(f"Verification failed: could not open collection: {exc}")
        return

    test_queries = [
        "Who leads the Robotics Research Lab?",
        "What are the current projects in the Robotics Research Lab?",
        "What is Ameca?",
        "What does RRLab research?",
        "What robots does RRLab have?",
        "Who is Ashita Ashok?",
    ]

    log("\nVerification queries")
    log("=" * 80)

    for original_query in test_queries:
        try:
            rewritten_query = rewrite_query_for_verification(original_query)
            inferred_category = infer_query_category(rewritten_query)

            log(f"\nQ: {original_query}")
            if rewritten_query != original_query:
                log(f"Rewritten: {rewritten_query}")

            all_rows = []

            if inferred_category:
                filtered = query_collection(collection, rewritten_query, inferred_category, top_k=VERIFY_TOP_K)
                all_rows.extend(flatten_results(filtered))
                print_hybrid_results(
                    f"Filtered + hybrid reranked where category='{inferred_category}'",
                    flatten_results(filtered),
                    rewritten_query,
                    inferred_category,
                )

            broad = query_collection(collection, rewritten_query, None, top_k=VERIFY_TOP_K)
            all_rows.extend(flatten_results(broad))
            print_hybrid_results("Broad + hybrid reranked", flatten_results(broad), rewritten_query, inferred_category)

            # Combined rerank is closest to what your main Self-RAG should do.
            unique = {}
            for row in all_rows:
                metadata = row["metadata"]
                key = (metadata.get("source", ""), metadata.get("chunk_index", ""))
                unique[key] = row

            print_hybrid_results("Combined final rerank", list(unique.values()), rewritten_query, inferred_category)

        except Exception as exc:
            log(f"   Verification query failed: {exc}")

    log("")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    pages = crawl_rrlab()

    if not pages:
        log("No usable pages extracted. Check network access and site structure.")
        sys.exit(1)

    chunks = build_chunks(pages)

    if not chunks:
        log("No usable chunks created. Check extraction/chunking settings.")
        sys.exit(1)

    build_knowledge_base(chunks)
    verify_kb()

    log("Knowledge base ready.")
    log("Start your main Ameca/VLM script and make sure it uses:")
    log(f"  CHROMA_PERSIST_DIR = '{CHROMA_PERSIST_DIR}'")
    log(f"  CHROMA_COLLECTION  = '{CHROMA_COLLECTION}'")
    log("")
    log("Recommended main-system retrieval settings:")
    log("  MAX_DISTANCE = 0.58")
    log("  use category filters from metadata when the query is about staff/projects/robots")
    log("  apply hybrid reranking: similarity + category bonus + priority bonus")
    log("  deduplicate final retrieved results by source URL")
    log("  penalize teaching pages when routing staff/project questions")
    log("  skip low-value archives/conference schedules/publication stubs during crawl")
    log("  canonicalize /en/seite to /en")
    log("  retrieve top 12, rerank, deduplicate by source, keep final 5")
    log("  use category caps to keep the vector DB balanced")


if __name__ == "__main__":
    main()
