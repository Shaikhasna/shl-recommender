"""
catalog.py
Loads the SHL product catalog JSON and provides search utilities.
No vector DB — the catalog is small enough to filter in memory and
pass relevant subsets directly to the LLM. Intentional design choice:
fewer moving parts, faster cold start, easier to defend in interviews.
"""

import json
import re
from pathlib import Path
from typing import Optional

# ── Load once at import time ──────────────────────────────────────────────────
_CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"


def _load_catalog() -> list[dict]:
    if not _CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {_CATALOG_PATH}. "
            "Download it from the SHL link and save as data/catalog.json"
        )
    with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        raw = json.loads(content, strict=False)

    # Normalize: accept either a list or a wrapped dict {"products": [...]}
    if isinstance(raw, list):
        return raw
    for key in ("products", "data", "items", "catalog"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    return list(raw.values())[0] if raw else []


CATALOG: list[dict] = _load_catalog()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text(item: dict) -> str:
    """Flatten all string fields of a catalog item into one searchable blob."""
    parts = []
    for v in item.values():
        if isinstance(v, str):
            parts.append(v.lower())
        elif isinstance(v, list):
            parts.extend(str(x).lower() for x in v)
    return " ".join(parts)


def _score(item: dict, keywords: list[str]) -> int:
    blob = _text(item)
    return sum(1 for kw in keywords if kw in blob)


# ── Public API ────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "for", "to", "of", "and", "or", "in", "is", "are",
    "we", "i", "with", "that", "this", "on", "need", "want", "use", "what",
    "how", "can", "should", "do", "be", "at", "it", "our", "their", "them",
    "us", "hiring", "hire", "looking", "assess", "assessment", "test", "my",
    "some", "also", "add", "any", "not", "no", "yes", "ok", "good", "please"
}


def search(query: str, top_k: int = 25) -> list[dict]:
    """
    Return up to top_k catalog items most relevant to query.
    Uses keyword overlap — fast, deterministic, fully debuggable.
    """
    tokens = [
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in STOPWORDS and len(w) > 2
    ]

    if not tokens:
        return CATALOG[:top_k]

    scored = [(item, _score(item, tokens)) for item in CATALOG]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = [item for item, s in scored if s > 0]
    return (results if results else CATALOG)[:top_k]


def get_by_name(name: str) -> Optional[dict]:
    name_lower = name.lower()
    for item in CATALOG:
        if name_lower in str(item.get("name", "")).lower():
            return item
    return None


def format_for_prompt(items: list[dict], max_items: int = 30) -> str:
    """Render catalog items as compact text block for the LLM prompt."""
    lines = []
    for item in items[:max_items]:
        name = item.get("name", "Unknown")
        url = item.get("url", item.get("link", ""))
        test_type = item.get("test_type", item.get("type", ""))
        duration = item.get("duration", item.get("duration_minutes", "—"))
        description = item.get("description", "")[:200]
        languages = item.get("languages", [])
        if isinstance(languages, list):
            lang_str = ", ".join(str(l) for l in languages[:5])
            if len(languages) > 5:
                lang_str += f" (+{len(languages)-5} more)"
        else:
            lang_str = str(languages)

        lines.append(
            f"- NAME: {name}\n"
            f"  TYPE: {test_type}\n"
            f"  DURATION: {duration}\n"
            f"  LANGUAGES: {lang_str}\n"
            f"  DESC: {description}\n"
            f"  URL: {url}"
        )
    return "\n\n".join(lines)


def catalog_summary() -> str:
    return f"SHL Individual Test Solutions catalog: {len(CATALOG)} assessments total."