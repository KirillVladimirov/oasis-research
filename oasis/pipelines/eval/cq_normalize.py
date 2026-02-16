"""Нормализация Competency Questions (CQ)."""
from __future__ import annotations

import re
from typing import List, Tuple

TOKEN_RE = re.compile(r"\b\w+\b")


def normalize_cq(text: str, *, remove_trailing_period: bool = False) -> Tuple[str, bool]:
    """Нормализует текст CQ для сопоставления и дедупликации."""
    normalized_text = text.strip()
    normalized_text = (
        normalized_text.replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    removed_trailing_period = False
    if remove_trailing_period and normalized_text.endswith(".") and "?" not in normalized_text:
        normalized_text = normalized_text[:-1]
        removed_trailing_period = True
    return normalized_text.lower(), removed_trailing_period


def tokenize(text: str) -> List[str]:
    """Разбивает текст CQ на токены."""
    return TOKEN_RE.findall(text)
