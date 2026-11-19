"""Unified Text Preprocessing Pipeline for AI Fake News Detection System.

Ensures 100% preprocessing consistency between offline training and online inference.
Implements:
1. Lowercase conversion
2. URL, email, and HTML stripping
3. Publisher / Wire Agency tag stripping (e.g. 'WASHINGTON (Reuters) - ...') to prevent data leakage
4. Non-alphanumeric character removal
5. Whitespace normalization
6. Stopword filtering (English)
7. NLTK WordNet lemmatization
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Patterns for data leakage prevention and text cleaning
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<.*?>")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Strip wire service headers and publisher/blog metadata leaks
_WIRE_SERVICE_PATTERN = re.compile(
    r"\b(?:[A-Za-z\s,]+)?(?:\(Reuters\)|\(AP\)|\(AFP\)|\(Bloomberg\)|\(Xinhua\))\s*[-–—]+|"
    r"\b(?:reuters|bloomberg|xinhua|breitbart|infowars|21wire|21st\s+century\s+wire)\b|"
    r"\b(?:featured\s+image|image\s+via|getty\s+images?|pic\.twitter\.com\S*|twitter\.com\S*)\b",
    re.IGNORECASE,
)

# Common metadata and publisher shortcuts that cause spurious dataset memorization
_SPURIOUS_TERMS: Set[str] = {
    "reuters", "ap", "afp", "bloomberg", "xinhua", "breitbart", "infowars", "21wire",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "image", "video", "watch", "photo", "picture", "featured", "getty", "com", "twitter", "pic", "wire", "21st", "century",
    "america", "american", "said", "via",
}

_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b")

_RESOURCES_INITIALIZED = False
_STOP_WORDS: Set[str] = set()
_LEMMATIZER: WordNetLemmatizer | None = None


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if missing."""
    global _RESOURCES_INITIALIZED, _STOP_WORDS, _LEMMATIZER

    if _RESOURCES_INITIALIZED and _LEMMATIZER is not None:
        return

    required = ["stopwords", "wordnet", "omw-1.4"]
    for res in required:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

    try:
        _STOP_WORDS = set(stopwords.words("english")).union(_SPURIOUS_TERMS)
    except Exception:
        _STOP_WORDS = set(_SPURIOUS_TERMS)

    _LEMMATIZER = WordNetLemmatizer()
    _RESOURCES_INITIALIZED = True


def clean_raw_text(text: str, strip_wire_tags: bool = True) -> str:
    """Clean raw text string: lowercasing, regex stripping, and whitespace normalization."""
    if not isinstance(text, str):
        text = str(text or "")

    text = text.lower()

    if strip_wire_tags:
        text = _WIRE_SERVICE_PATTERN.sub(" ", text)

    text = _URL_PATTERN.sub(" ", text)
    text = _HTML_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def preprocess_text(text: str, strip_wire_tags: bool = True) -> str:
    """Full preprocessing pipeline: clean, tokenize, filter stopwords, and lemmatize."""
    ensure_nltk_resources()
    lemmatizer = _LEMMATIZER or WordNetLemmatizer()
    stop_words = _STOP_WORDS

    cleaned = clean_raw_text(text, strip_wire_tags=strip_wire_tags)
    tokens = _TOKEN_PATTERN.findall(cleaned)

    # Filter stopwords and short tokens (len <= 2), apply lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(processed_tokens)
