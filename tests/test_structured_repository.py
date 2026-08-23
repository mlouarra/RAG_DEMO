import json

import pytest

from fm_rag.structured_repository import (
    calculate_series_evolution,
    load_structured_payload,
    query_measure_evolution,
    select_measure_series,
)


def build_measure(
    period: str,
    value: int,
    file_name: str,
) -> dict:
    return {
        "entity_name": "ENSEIGNES PRESSE",
        "indicator": "outlet_count",
        "unit": "count",
        "observation_period": period,
        "value": value,
        "file_name": file_name,
        "page_number": 3,
        "source_bbox": [180.0, 460.0, 250.0, 480.0],
    }


def test_load_structured_payload(tmp_path) -> None:
    json_path = tmp_path / "measures.json"

    payload = {
        "schema_version": "1.0",
        "document_count": 2,
        "measure_count": 2,
        "measures": [
            build_measure(
                "2021-12",
                2364,
                "rapport_2021.pdf",
            ),
            build_measure(
                "2026-06",
                1959,
                "rapport_2026.pdf",
            ),
        ],
    }

    json_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded_payload = load_structured_payload(json_path)

    assert loaded_payload["schema_version"] == "1.0"
    assert loaded_payload["document_count"] == 2
    assert loaded_payload["measure_count"] == 2
    assert len(loaded_payload["measures"]) == 2


def test_load_structured_payload_rejects_wrong_count(
    tmp_path,
) -> None:
    json_path = tmp_path / "measures.json"

    payload = {
        "schema_version": "1.0",
        "document_count": 1,
        "measure_count": 10,
        "measures": [],
    }

    json_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Nombre de mesures incohérent",
    ):
        load_structured_payload(json_path)


def test_select_measure_series_sorts_periods() -> None:
    payload = {
        "measures": [
            build_measure(
                "2026-06",
                1959,
                "rapport_2026.pdf",
            ),
            build_measure(
                "2021-12",
                2364,
                "rapport_2021.pdf",
            ),
        ]
    }

    measures = select_measure_series(
        payload=payload,
        entity_name="enseignes presse",
        indicator="outlet_count",
    )

    assert len(measures) == 2
    assert measures[0]["observation_period"] == "2021-12"
    assert measures[1]["observation_period"] == "2026-06"


def test_calculate_series_evolution() -> None:
    measures = [
        build_measure(
            "2021-12",
            2364,
            "rapport_2021.pdf",
        ),
        build_measure(
            "2026-06",
            1959,
            "rapport_2026.pdf",
        ),
    ]

    evolution = calculate_series_evolution(measures)

    assert evolution["start_period"] == "2021-12"
    assert evolution["end_period"] == "2026-06"
    assert evolution["start_value"] == 2364
    assert evolution["end_value"] == 1959
    assert evolution["absolute_change"] == -405
    assert evolution["percentage_change"] == pytest.approx(-17.1319796954)
    assert evolution["start_source"]["file_name"] == "rapport_2021.pdf"
    assert evolution["end_source"]["file_name"] == "rapport_2026.pdf"


def test_select_measure_series_filters_periods() -> None:
    payload = {
        "measures": [
            build_measure(
                "2021-12",
                2364,
                "rapport_2021.pdf",
            ),
            build_measure(
                "2022-12",
                2341,
                "rapport_2022.pdf",
            ),
            build_measure(
                "2023-12",
                2337,
                "rapport_2023.pdf",
            ),
            build_measure(
                "2024-12",
                2045,
                "rapport_2024.pdf",
            ),
            build_measure(
                "2025-12",
                1998,
                "rapport_2025.pdf",
            ),
        ]
    }

    measures = select_measure_series(
        payload=payload,
        entity_name="ENSEIGNES PRESSE",
        indicator="outlet_count",
        start_period="2022-12",
        end_period="2024-12",
    )

    assert [measure["observation_period"] for measure in measures] == [
        "2022-12",
        "2023-12",
        "2024-12",
    ]


def test_select_measure_series_rejects_invalid_period() -> None:
    payload = {"measures": []}

    with pytest.raises(
        ValueError,
        match="Période invalide",
    ):
        select_measure_series(
            payload=payload,
            entity_name="ENSEIGNES PRESSE",
            indicator="outlet_count",
            start_period="décembre 2021",
        )


def test_select_measure_series_rejects_reversed_periods() -> None:
    payload = {"measures": []}

    with pytest.raises(
        ValueError,
        match="La période de début",
    ):
        select_measure_series(
            payload=payload,
            entity_name="ENSEIGNES PRESSE",
            indicator="outlet_count",
            start_period="2025-12",
            end_period="2021-12",
        )


def test_query_measure_evolution() -> None:
    payload = {
        "measures": [
            build_measure(
                "2021-12",
                2364,
                "rapport_2021.pdf",
            ),
            build_measure(
                "2026-06",
                1959,
                "rapport_2026.pdf",
            ),
        ]
    }

    result = query_measure_evolution(
        payload=payload,
        entity_name="ENSEIGNES PRESSE",
        indicator="outlet_count",
        start_period="2021-12",
        end_period="2026-06",
    )

    assert result["measure_count"] == 2
    assert result["evolution"]["absolute_change"] == -405
    assert result["query"]["entity_name"] == "ENSEIGNES PRESSE"


def test_query_measure_evolution_rejects_empty_result() -> None:
    payload = {"measures": []}

    with pytest.raises(
        ValueError,
        match="Aucune mesure trouvée",
    ):
        query_measure_evolution(
            payload=payload,
            entity_name="KIOSQUES",
            indicator="outlet_count",
        )


def test_query_measure_evolution_rejects_single_measure() -> None:
    payload = {
        "measures": [
            build_measure(
                "2021-12",
                2364,
                "rapport_2021.pdf",
            )
        ]
    }

    with pytest.raises(
        ValueError,
        match="Une seule mesure",
    ):
        query_measure_evolution(
            payload=payload,
            entity_name="ENSEIGNES PRESSE",
            indicator="outlet_count",
        )
