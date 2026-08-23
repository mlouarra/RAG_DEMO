from pathlib import Path

import pytest

from fm_rag.structured_extraction import (
    audit_additive_totals,
    extract_outlet_distribution_measures,
    extract_outlet_type_measures,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

PDF_PATHS = sorted(DATA_DIR.glob("*.pdf"))

EXPECTED_OUTLET_COUNTS = {
    "2021-12": 2364,
    "2022-12": 2341,
    "2023-12": 2337,
    "2024-12": 2045,
    "2025-12": 1998,
    "2026-06": 1959,
}


@pytest.mark.skipif(
    len(PDF_PATHS) != 6,
    reason="Les six PDF France Messagerie sont nécessaires.",
)
def test_extract_enseignes_presse_from_all_reports() -> None:
    extracted_counts: dict[str, int | float] = {}

    for pdf_path in PDF_PATHS:
        measures = extract_outlet_type_measures(
            pdf_path=pdf_path,
            entity_name="ENSEIGNES PRESSE",
        )

        assert len(measures) == 7

        outlet_count = next(measure for measure in measures if measure.indicator == "outlet_count")

        assert outlet_count.page_number == 3
        assert outlet_count.entity_name == "ENSEIGNES PRESSE"
        assert outlet_count.unit == "count"

        extracted_counts[outlet_count.observation_period] = outlet_count.value

    assert extracted_counts == EXPECTED_OUTLET_COUNTS


@pytest.mark.skipif(
    len(PDF_PATHS) != 6,
    reason="Les six PDF France Messagerie sont nécessaires.",
)
def test_validate_additive_totals_for_all_reports() -> None:
    reports_by_year = {}

    for pdf_path in PDF_PATHS:
        measures = extract_outlet_distribution_measures(pdf_path)

        report = audit_additive_totals(measures)

        year = measures[0].observation_period[:4]

        reports_by_year[year] = report

    # Aucun rapport ne contient d'erreur bloquante.
    assert all(report.is_valid for report in reports_by_year.values())

    assert all(report.errors == () for report in reports_by_year.values())

    # Seul le rapport 2024 contient un avertissement.
    warning_years = {year for year, report in reports_by_year.items() if report.warnings}

    assert warning_years == {"2024"}

    assert reports_by_year["2024"].warnings == (
        "linear_km: somme=1560, total affiché=1559, écart=1 (écart d'arrondi toléré)",
    )
