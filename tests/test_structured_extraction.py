import pytest

from fm_rag.structured_extraction import (
    normalize_text,
    parse_decimal,
    parse_integer,
    parse_percentage,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2364", 2364),
        ("2 364", 2364),
        ("-47", -47),
        ("\u221247", -47),
    ],
)
def test_parse_integer(
    raw_value: str,
    expected: int,
) -> None:
    assert parse_integer(raw_value) == expected


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("332", 332.0),
        ("12,7", 12.7),
        ("-11,9", -11.9),
        ("\u221211,9", -11.9),
    ],
)
def test_parse_decimal(
    raw_value: str,
    expected: float,
) -> None:
    assert parse_decimal(raw_value) == expected


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("7,1%", 7.1),
        ("-2,3%", -2.3),
        ("\u22124,6%", -4.6),
    ],
)
def test_parse_percentage(
    raw_value: str,
    expected: float,
) -> None:
    assert parse_percentage(raw_value) == expected


def test_parse_percentage_rejects_missing_percent_sign() -> None:
    with pytest.raises(
        ValueError,
        match="Pourcentage invalide",
    ):
        parse_percentage("7,1")


def test_normalize_text_removes_accents_and_case() -> None:
    result = normalize_text("Répartition et Évolution")

    assert result == "REPARTITION ET EVOLUTION"
