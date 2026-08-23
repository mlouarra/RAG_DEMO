from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fm_rag.structured_extraction import (
    audit_additive_totals,
    extract_kiosk_measures,
    extract_outlet_distribution_measures,
)


DOCUMENTS_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/structured")

OUTLET_OUTPUT_PATH = OUTPUT_DIR / "outlet_distribution.json"
KIOSK_OUTPUT_PATH = OUTPUT_DIR / "kiosk_evolution.json"


def determine_status(
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
) -> str:
    if errors:
        return "invalid"

    if warnings:
        return "valid_with_warnings"

    return "valid"


def main() -> None:
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(f"Aucun PDF trouvé dans {DOCUMENTS_DIR}.")

    all_measures = []
    all_kiosk_measures = []
    validation_results = []

    for pdf_path in pdf_paths:
        measures = extract_outlet_distribution_measures(pdf_path)

        validation_report = audit_additive_totals(measures)

        all_measures.extend(measures)

        validation_results.append(
            {
                "file_name": pdf_path.name,
                "status": determine_status(
                    errors=validation_report.errors,
                    warnings=validation_report.warnings,
                ),
                "errors": list(validation_report.errors),
                "warnings": list(validation_report.warnings),
            }
        )

        kiosk_measures = extract_kiosk_measures(pdf_path)

        all_kiosk_measures.extend(kiosk_measures)

    outlet_payload = {
        "schema_version": "1.0",
        "document_count": len(pdf_paths),
        "measure_count": len(all_measures),
        "measures": [
            asdict(measure)
            for measure in all_measures
        ],
        "validation": validation_results,
    }

    kiosk_payload = {
        "schema_version": "1.0",
        "document_count": len(pdf_paths),
        "measure_count": len(all_kiosk_measures),
        "measures": [
            asdict(measure)
            for measure in all_kiosk_measures
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTLET_OUTPUT_PATH.write_text(
        json.dumps(
            outlet_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    KIOSK_OUTPUT_PATH.write_text(
        json.dumps(
            kiosk_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Documents traités          : {len(pdf_paths)}")
    print(f"Mesures distribution       : {len(all_measures)}")
    print(f"Mesures kiosques           : {len(all_kiosk_measures)}")
    print(f"Fichier distribution       : {OUTLET_OUTPUT_PATH}")
    print(f"Fichier kiosques           : {KIOSK_OUTPUT_PATH}")

    invalid_count = sum(
        result["status"] == "invalid"
        for result in validation_results
    )

    warning_count = sum(
        result["status"] == "valid_with_warnings"
        for result in validation_results
    )

    print(f"Rapports invalides             : {invalid_count}")
    print(f"Rapports avec avertissement    : {warning_count}")


if __name__ == "__main__":
    main()