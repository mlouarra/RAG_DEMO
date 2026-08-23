from __future__ import annotations

import re

from fm_rag.schemas import DocumentPage, TextChunk

PAGE_SECTIONS = {
    1: "page_de_garde",
    2: "evolution_effectif",
    3: "repartition_types_enseigne",
    4: "evolution_pdv_ca",
    5: "entrants_sortants",
    6: "segmentation_lineaire",
}


def clean_page_text(text: str) -> str:
    lines = []

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def build_page_chunk(page: DocumentPage) -> TextChunk | None:
    cleaned_text = clean_page_text(page.text)

    if not cleaned_text:
        return None

    report_month = page.metadata.get("report_month")
    report_year = page.metadata.get("report_year")
    section = PAGE_SECTIONS.get(page.page_number, "autre")

    context_header = (
        f"Document : {page.file_name}\n"
        f"Période : {report_month} {report_year}\n"
        f"Page : {page.page_number}/{page.page_count}\n"
        f"Section : {section}\n"
    )

    chunk_text = f"{context_header}\n{cleaned_text}"

    metadata = {
        **page.metadata,
        "file_name": page.file_name,
        "section": section,
        "content_type": "page",
        "chunking_strategy": "one_chunk_per_page",
    }

    return TextChunk(
        document_id=page.document_id,
        chunk_id=f"{page.document_id}_page_{page.page_number:03d}",
        text=chunk_text,
        metadata=metadata,
    )


def build_chunks(pages: list[DocumentPage]) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for page in pages:
        chunk = build_page_chunk(page)

        if chunk is not None:
            chunks.append(chunk)

    return chunks
