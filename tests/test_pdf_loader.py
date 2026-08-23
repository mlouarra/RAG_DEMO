from pathlib import Path

import fitz
import pytest

from fm_rag.pdf_loader import (
    extract_report_metadata,
    load_all_pdf_pages,
    load_pdf_pages,
)


def create_test_pdf(path: Path) -> None:
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "Suivi du réseau des marchands de presse - décembre 2025",
    )

    second_page = document.new_page()
    second_page.insert_text(
        (72, 72),
        "Le réseau compte 18 473 points de vente actifs.",
    )

    document.save(path)
    document.close()


def test_extract_report_metadata() -> None:
    metadata = extract_report_metadata("Suivi_reseau_synthèse_décembre_2025.pdf")

    assert metadata["report_year"] == 2025
    assert metadata["report_month"] == "decembre"


def test_load_pdf_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "suivi_reseau_decembre_2025.pdf"
    create_test_pdf(pdf_path)

    pages = load_pdf_pages(pdf_path)

    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert pages[0].page_count == 2
    assert pages[0].metadata["report_year"] == 2025
    assert pages[0].metadata["report_month"] == "decembre"

    assert pages[1].page_number == 2
    assert "18 473 points de vente" in pages[1].text


def test_load_all_pdf_pages(tmp_path: Path) -> None:
    create_test_pdf(tmp_path / "rapport_decembre_2024.pdf")
    create_test_pdf(tmp_path / "rapport_decembre_2025.pdf")

    pages = load_all_pdf_pages(tmp_path)

    assert len(pages) == 4
    assert {page.metadata["report_year"] for page in pages} == {
        2024,
        2025,
    }


def test_load_pdf_pages_rejects_non_pdf(tmp_path: Path) -> None:
    text_path = tmp_path / "document.txt"
    text_path.write_text("Contenu", encoding="utf-8")

    with pytest.raises(ValueError, match="n'est pas un PDF"):
        load_pdf_pages(text_path)
