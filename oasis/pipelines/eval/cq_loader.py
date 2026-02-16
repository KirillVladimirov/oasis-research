# Загрузка CQ из JSONL с валидацией (CQRecord, load_cqs).

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TEXT_KEYS = ("cq", "question", "text")
ID_KEYS = ("id", "cq_id")


class CQRecord(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    id: str = Field(..., description="Уникальный идентификатор")
    text: str = Field(..., description="Текст CQ")
    topic: str | None = None
    cluster: str | None = None
    source: str | None = None
    model: str | None = None
    prompt_id: str | None = None
    kept: bool | None = None

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty")
        return value


class CQLoadError(ValueError):
    pass


def _pick_text(obj: dict) -> tuple[str, str]:
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str):
            return value, key
    return "", ""


def _pick_id(obj: dict) -> str | None:
    for key in ID_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _hash_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"cq_{digest[:12]}"


def normalize_record(obj: dict) -> dict[str, Any]:
    text, _ = _pick_text(obj)
    if not text:
        raise CQLoadError("missing text field")
    record = dict(obj)
    record["text"] = text
    record["id"] = _pick_id(obj) or _hash_id(text)
    return record


def load_cqs(path: Path) -> list[CQRecord]:
    records: list[CQRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CQLoadError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise CQLoadError(f"line {line_no}: expected JSON object")
            try:
                record = CQRecord.model_validate(normalize_record(obj))
            except (CQLoadError, ValidationError) as exc:
                raise CQLoadError(f"line {line_no}: {exc}") from exc
            records.append(record)
    return records
