from __future__ import annotations

import io

from PIL import Image

from maintenance_triage_copilot.domain.models import CorpusDocument
from maintenance_triage_copilot.retrieval import document_parser


def _png_bytes() -> bytes:
    image = Image.new("RGB", (24, 24), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_parse_image_bytes_uses_ocr(monkeypatch) -> None:
    monkeypatch.setattr(
        document_parser,
        "_ocr_pil_image",
        lambda image: "Breaker B4 shows red fault lamp.",
    )

    document = document_parser.parse_image_bytes(
        _png_bytes(),
        document_id="img-doc",
        title="Panel Snapshot",
    )

    assert document.document_id == "img-doc"
    assert document.title == "Panel Snapshot"
    assert document.body == "Breaker B4 shows red fault lamp."


def test_parse_pdf_uses_ocr_when_embedded_text_is_sparse(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(document_parser, "_extract_pdf_text_embedded", lambda path: "short text")
    monkeypatch.setattr(
        document_parser,
        "_extract_pdf_text_via_ocr",
        lambda path: "OCR recovered maintenance procedure.",
    )

    document = document_parser.parse_pdf(pdf_path, document_id="pdf-doc", title="OCR Manual")

    assert document.document_id == "pdf-doc"
    assert document.body == "OCR recovered maintenance procedure."


def test_corpus_upload_accepts_image_documents(client, monkeypatch) -> None:
    monkeypatch.setattr(
        document_parser,
        "parse_image_bytes",
        lambda data, document_id, **kwargs: CorpusDocument(
            document_id=document_id,
            source_type=kwargs["source_type"],
            title=kwargs["title"],
            body="OCR extracted inspection checklist.",
            equipment_family=kwargs["equipment_family"],
            tags=kwargs["tags"],
        ),
    )

    response = client.post(
        "/corpus/upload",
        files={"file": ("panel.png", _png_bytes(), "image/png")},
        data={
            "document_id": "upload-img-doc",
            "source_type": "manual",
            "title": "Uploaded Panel Photo",
            "tags": "fault_light_on",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "indexed"
    assert payload["document_id"] == "upload-img-doc"
