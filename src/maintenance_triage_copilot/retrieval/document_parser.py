"""Document parsers for ingesting maintenance manuals, SOPs, and tickets.

Handles PDF, plain text, and Markdown files. Extracts clean text for
chunking and vector indexing.
"""

from __future__ import annotations

import re
from pathlib import Path

from maintenance_triage_copilot.domain.models import CorpusDocument, CorpusSourceType


def parse_pdf(path: str | Path, document_id: str, **kwargs) -> CorpusDocument:
    """Parse a PDF file into a CorpusDocument.

    Requires PyMuPDF (fitz) or pdfplumber. Falls back gracefully.
    """
    path = Path(path)
    text = _extract_pdf_text(path)
    return CorpusDocument(
        document_id=document_id,
        source_type=kwargs.get("source_type", CorpusSourceType.manual),
        title=kwargs.get("title", path.stem),
        body=text,
        equipment_family=kwargs.get("equipment_family", "electrical_panel_family_a"),
        tags=kwargs.get("tags", []),
    )


def parse_text_file(path: str | Path, document_id: str, **kwargs) -> CorpusDocument:
    """Parse a plain text or Markdown file into a CorpusDocument."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    # Strip markdown headers/formatting for cleaner retrieval
    text = _clean_markdown(text)
    return CorpusDocument(
        document_id=document_id,
        source_type=kwargs.get("source_type", CorpusSourceType.manual),
        title=kwargs.get("title", path.stem),
        body=text,
        equipment_family=kwargs.get("equipment_family", "electrical_panel_family_a"),
        tags=kwargs.get("tags", []),
    )


def parse_file(path: str | Path, document_id: str, **kwargs) -> CorpusDocument:
    """Auto-detect file type and parse accordingly."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, document_id, **kwargs)
    if suffix in {".txt", ".md", ".rst"}:
        return parse_text_file(path, document_id, **kwargs)
    raise ValueError(f"Unsupported file type: {suffix}. Expected .pdf, .txt, or .md")


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, trying multiple backends."""
    # Try PyMuPDF first (fastest, best quality)
    try:
        import fitz

        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass

    raise ImportError(
        "PDF parsing requires either PyMuPDF or pdfplumber. "
        "Install with: pip install PyMuPDF  OR  pip install pdfplumber"
    )


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting for cleaner text retrieval."""
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove link syntax, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
