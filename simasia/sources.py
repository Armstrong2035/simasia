"""Turn training URLs into a single text corpus.

Fetching a page is easy; extracting *clean* prose (no nav, cookie banners,
footers, or ads) is the hard part, so this delegates both steps to
``trafilatura`` rather than hand-rolling an HTML reader.  Each URL's extracted
article text is concatenated into one corpus, which the guard then chunks
normally — a page's content is raw material, never a single chunk.
"""

from __future__ import annotations

import re
from typing import Callable

CORPUS_SEPARATOR = "\n\n"

# Patterns for boilerplate that pollutes brand voice: URLs, emails, star glyphs,
# "Source:" lines, "N min read", dates, e-commerce page-reload notices, and
# social/contact handles. Each is replaced with a space.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_CLEANERS = [
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"[★☆⭐]"),
    re.compile(r"(?im)^\s*source:.*$"),
    re.compile(r"(?i)\b\d+\s*min read\b"),
    re.compile(rf"(?i)\b(?:{_MONTHS})\s+\d{{1,2}},\s*\d{{4}}\b"),
    re.compile(r"(?i)[^.!?]*\bpage (?:refresh|reload)[^.!?]*[.!?]"),
    re.compile(r"(?i)\b(?:instagram|whatsapp|email)\s*:\s*\S*"),
    re.compile(r"[•·▪]"),
]


def clean_corpus(text: str) -> str:
    """Strip common web boilerplate so only brand voice remains, then normalise
    whitespace. Applied to training corpora before chunking."""
    for pattern in _CLEANERS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_url_text(url: str) -> str:
    """Download ``url`` and return its main article text.

    Raises ``ValueError`` when the page cannot be downloaded or yields no
    extractable content, so a dead link never silently contributes an empty
    sample to training.
    """
    try:
        import trafilatura
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "URL training requires the 'trafilatura' package. "
            "Install it with: pip install simasia[urls]"
        ) from exc

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(f"Could not download URL: {url!r}")
    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        raise ValueError(f"No extractable text content at URL: {url!r}")
    return text.strip()


def build_corpus(
    urls: list[str],
    fetcher: Callable[[str], str] = fetch_url_text,
) -> str:
    """Fetch every URL and join the results into one corpus string.

    ``fetcher`` is injectable so callers can supply cached content or a stub in
    tests without network access.
    """
    if not urls:
        raise ValueError("At least one URL is required.")
    return CORPUS_SEPARATOR.join(fetcher(url) for url in urls)
