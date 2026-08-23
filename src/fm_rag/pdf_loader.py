from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf

from fm_rag.schemas import DocumentPage

MONTH_NAMES = (
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_accents.lower()


def build_document_id(pdf_path: Path) -> str:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:12]

    normalized_stem = normalize_text(pdf_path.stem)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_stem).strip("-")

    return f"{slug}_{digest}"


def extract_report_metadata(file_name: str) -> dict[str, Any]:
    normalized_name = normalize_text(Path(file_name).stem)

    year_match = re.search(r"(20\d{2})", normalized_name)
    report_year = int(year_match.group(1)) if year_match else None

    report_month = next(
        (month for month in MONTH_NAMES if month in normalized_name),
        None,
    )

    return {
        "report_year": report_year,
        "report_month": report_month,
    }


def load_pdf_pages(
    pdf_path: str | Path,
    minimum_text_length: int = 20,
) -> list[DocumentPage]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable : {path}")

    if not path.is_file():
        raise ValueError(f"Le chemin ne correspond pas à un fichier : {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Le fichier n'est pas un PDF : {path}")

    document_id = build_document_id(path)
    report_metadata = extract_report_metadata(path.name)

    pages: list[DocumentPage] = []

    with pymupdf.open(path) as pdf_document:
        page_count = pdf_document.page_count

        for page_index, pdf_page in enumerate(pdf_document):
            page_number = page_index + 1
            text = pdf_page.get_text("text").strip()

            metadata = {
                **report_metadata,
                "page_number": page_number,
                "page_count": page_count,
                "source_type": "pdf",
                "text_length": len(text),
                "has_sufficient_text": len(text) >= minimum_text_length,
            }

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    file_name=path.name,
                    file_path=path,
                    page_number=page_number,
                    page_count=page_count,
                    text=text,
                    metadata=metadata,
                )
            )

    return pages


def load_all_pdf_pages(
    documents_dir: str | Path,
    minimum_text_length: int = 20,
) -> list[DocumentPage]:
    directory = Path(documents_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Répertoire des documents introuvable : {directory}")

    pages: list[DocumentPage] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        pages.extend(
            load_pdf_pages(
                pdf_path=pdf_path,
                minimum_text_length=minimum_text_length,
            )
        )

    return pages
