"""Document chunking."""

from __future__ import annotations

import re

from maintenance_triage_copilot.domain.models import CorpusDocument

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|step\s+\d+[:.)]?|[a-z][.)])\s*", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")


def chunk_document(document: CorpusDocument, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Create overlap-aware chunks while respecting sentence and list boundaries."""
    units = _document_units(document.body)
    if not units:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    index = 0
    current_units: list[str] = []
    current_len = 0
    seen_texts: set[str] = set()

    while index < len(units):
        unit = units[index]
        proposed = current_len + len(unit) + (1 if current_units else 0)
        if current_units and proposed > chunk_size:
            body = " ".join(current_units).strip()
            if body and body not in seen_texts:
                chunks.append(
                    {
                        "chunk_id": f"{document.document_id}-chunk-{chunk_index}",
                        "document_id": document.document_id,
                        "title": document.title,
                        "source_type": document.source_type,
                        "equipment_family": document.equipment_family,
                        "text": body,
                        "tags": list(document.tags),
                    }
                )
                seen_texts.add(body)
                chunk_index += 1
            overlap_units = _tail_overlap(current_units, chunk_overlap)
            if overlap_units == current_units:
                overlap_units = current_units[1:]
            current_units = overlap_units
            current_len = _joined_len(current_units)
            continue

        current_units.append(unit)
        current_len = proposed
        index += 1

    body = " ".join(current_units).strip()
    if body and body not in seen_texts:
        chunks.append(
            {
                "chunk_id": f"{document.document_id}-chunk-{chunk_index}",
                "document_id": document.document_id,
                "title": document.title,
                "source_type": document.source_type,
                "equipment_family": document.equipment_family,
                "text": body,
                "tags": list(document.tags),
            }
        )
    return chunks


def _document_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    units: list[str] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if any(_LIST_PREFIX_RE.match(line) for line in lines):
            for line in lines:
                cleaned = _clean_line(line)
                if cleaned:
                    units.extend(_split_large_unit(cleaned))
            continue
        paragraph = " ".join(lines)
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            cleaned = _clean_line(sentence)
            if cleaned:
                units.extend(_split_large_unit(cleaned))
    return units


def _clean_line(text: str) -> str:
    cleaned = _LIST_PREFIX_RE.sub("", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _split_large_unit(text: str, limit: int = 220) -> list[str]:
    if len(text) <= limit:
        return [text]

    clause_split = re.split(
        (
            r"(?<=,)\s+"
            r"(?=(?:then|after|before|if|verify|inspect|check|measure|tighten|reset|"
            r"replace|confirm|escalate)\b)"
        ),
        text,
        flags=re.IGNORECASE,
    )
    parts = [part.strip() for part in clause_split if part.strip()]
    if len(parts) > 1:
        return parts

    return [
        text[idx : idx + limit].strip()
        for idx in range(0, len(text), limit)
        if text[idx : idx + limit].strip()
    ]


def _tail_overlap(units: list[str], chunk_overlap: int) -> list[str]:
    if not units or chunk_overlap <= 0:
        return []

    selected: list[str] = []
    total = 0
    for unit in reversed(units):
        selected.insert(0, unit)
        total += len(unit) + (1 if selected else 0)
        if total >= chunk_overlap:
            break
    return selected


def _joined_len(units: list[str]) -> int:
    if not units:
        return 0
    return sum(len(unit) for unit in units) + max(0, len(units) - 1)
