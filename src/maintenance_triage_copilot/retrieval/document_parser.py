"""Document parsers for ingesting maintenance manuals, SOPs, and tickets.

Handles PDF, plain text, and Markdown files. Extracts clean text for
chunking and vector indexing.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import cast

from maintenance_triage_copilot.domain.models import CorpusDocument, CorpusSourceType

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def parse_pdf(path: str | Path, document_id: str, **kwargs) -> CorpusDocument:
    """Parse a PDF file into a CorpusDocument.

    Requires PyMuPDF (fitz) or pdfplumber. OCR is used when embedded text is
    missing or sparse.
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


def parse_image_file(path: str | Path, document_id: str, **kwargs) -> CorpusDocument:
    """Parse an image document via OCR."""
    path = Path(path)
    text = _extract_image_text(path)
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
    if suffix in _IMAGE_SUFFIXES:
        return parse_image_file(path, document_id, **kwargs)
    raise ValueError(
        f"Unsupported file type: {suffix}. Expected .pdf, .txt, .md, or OCR-compatible images"
    )


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF, using OCR when embedded text is insufficient."""
    embedded_text = _extract_pdf_text_embedded(path)
    if not _needs_ocr(embedded_text):
        return embedded_text

    try:
        ocr_text = _extract_pdf_text_via_ocr(path)
    except ImportError:
        ocr_text = ""
    if ocr_text.strip():
        return ocr_text
    return embedded_text


def _extract_pdf_text_embedded(path: Path) -> str:
    """Extract embedded text from a PDF, trying multiple backends."""
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


def _extract_pdf_text_via_ocr(path: Path) -> str:
    """Render PDF pages to images and OCR them."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PDF OCR requires PyMuPDF") from exc

    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            image = _pil_image_from_bytes(pixmap.tobytes("png"))
            pages.append(_ocr_pil_image(image))
    return "\n\n".join(page for page in pages if page.strip()).strip()


def parse_image_bytes(data: bytes, document_id: str, **kwargs) -> CorpusDocument:
    """Parse image bytes via OCR."""
    text = _ocr_pil_image(_pil_image_from_bytes(data))
    return CorpusDocument(
        document_id=document_id,
        source_type=kwargs.get("source_type", CorpusSourceType.manual),
        title=kwargs.get("title", "ocr-document"),
        body=text,
        equipment_family=kwargs.get("equipment_family", "electrical_panel_family_a"),
        tags=kwargs.get("tags", []),
    )


def _extract_image_text(path: Path) -> str:
    return _ocr_pil_image(_pil_image_from_bytes(path.read_bytes()))


def _pil_image_from_bytes(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def _ocr_pil_image(image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError("OCR requires pytesseract to be installed") from exc

    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        raise RuntimeError("OCR failed; verify that the tesseract binary is installed") from exc
    return cast(str, text).strip()


def _needs_ocr(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < 40


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
