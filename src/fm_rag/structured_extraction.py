from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from fm_rag.pdf_loader import (
    build_document_id,
    extract_report_metadata,
)
from fm_rag.schemas import StructuredMeasure

NumericValue = int | float
PdfWord = tuple[Any, ...]
BoundingBox = tuple[float, float, float, float]

OUTLET_TYPE_ENTITIES = (
    "RESEAU TRADITIONNEL",
    "ENSEIGNES PRESSE",
    "ENSEIGNES NON PRESSE",
    "AUTRES",
    "TOTAL",
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    zone: tuple[float, float]
    indicator: str
    unit: str
    parser: Callable[[str], NumericValue]
    is_comparison: bool


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

@dataclass(frozen=True, slots=True)
class NumericRowValues:
    comparison_count: int
    observed_count: int
    absolute_change: int
    percentage_change: float
    revenue_change_percentage: float | None


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFKD",
        value,
    )

    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

    return " ".join(without_accents.upper().split())


def normalize_number_text(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "")
        .replace("\u2212", "-")
        .replace(",", ".")
    )


def is_numeric_token(value: str) -> bool:
    normalized_value = normalize_number_text(value)

    if normalized_value.endswith("%"):
        normalized_value = normalized_value[:-1]

    try:
        float(normalized_value)
    except ValueError:
        return False

    return True

def parse_integer(value: str) -> int:
    return int(normalize_number_text(value))


def parse_decimal(value: str) -> float:
    return float(normalize_number_text(value))


def parse_percentage(value: str) -> float:
    stripped_value = value.strip()

    if not stripped_value.endswith("%"):
        raise ValueError(f"Pourcentage invalide : {value!r}")

    without_percent = stripped_value.removesuffix("%")

    return float(normalize_number_text(without_percent))


def calculate_bbox(
    words: list[PdfWord],
) -> BoundingBox:
    if not words:
        raise ValueError("Impossible de calculer une boîte vide.")

    return (
        min(float(word[0]) for word in words),
        min(float(word[1]) for word in words),
        max(float(word[2]) for word in words),
        max(float(word[3]) for word in words),
    )


def extract_cell(
    words: list[PdfWord],
    zone: tuple[float, float],
    target_y: float,
    tolerance: float,
) -> tuple[str, BoundingBox]:
    x_min, x_max = zone

    cell_words = [
        word
        for word in words
        if (
            abs(float(word[1]) - target_y) <= tolerance
            and x_min <= float(word[0]) < x_max
        )
    ]

    if not cell_words:
        raise ValueError(f"Cellule vide dans la zone {zone}.")

    cell_words.sort(key=lambda word: float(word[0]))

    raw_value = "".join(str(word[4]) for word in cell_words)

    return raw_value, calculate_bbox(cell_words)


def find_page_by_title(
    document: pymupdf.Document,
    expected_title: str,
) -> int:
    normalized_title = normalize_text(expected_title)

    for page_index, page in enumerate(document):
        normalized_page_text = normalize_text(page.get_text("text"))

        if normalized_title in normalized_page_text:
            return page_index

    raise ValueError(f"Page introuvable : {expected_title!r}")


def find_row_y(
    words: list[PdfWord],
    expected_label: str,
    label_x_max: float = 190.0,
    table_y_min: float = 430.0,
) -> float:
    lines: dict[float, list[PdfWord]] = {}

    for word in words:
        x0 = float(word[0])
        y0 = float(word[1])

        if x0 < label_x_max and y0 >= table_y_min:
            rounded_y = round(y0, 1)
            lines.setdefault(
                rounded_y,
                [],
            ).append(word)

    normalized_expected_label = normalize_text(expected_label)

    for y, line_words in lines.items():
        line_words.sort(key=lambda word: float(word[0]))

        label = " ".join(str(word[4]) for word in line_words)

        if normalize_text(label) == normalized_expected_label:
            return y

    raise ValueError(f"Ligne introuvable : {expected_label!r}")

KIOSK_PAGE_TYPE = "kiosk_evolution"

KIOSK_ENTITY_TYPE = "outlet_subtype"

KIOSK_ENTITY_NAME = "Kiosques"

def build_kiosk_measures(
    values: NumericRowValues,
    numeric_words: list[PdfWord],
    document_id: str,
    file_name: str,
    page_number: int,
    observation_period: str,
    comparison_period: str,
) -> list[StructuredMeasure]:
    if len(numeric_words) not in (4, 5):
        raise ValueError(
            f"Nombre inattendu de tokens numériques : {len(numeric_words)}"
        )

    observed_word = numeric_words[1]
    absolute_change_word = numeric_words[2]
    percentage_change_word = numeric_words[3]

    measures = [
        StructuredMeasure(
            document_id=document_id,
            file_name=file_name,
            page_number=page_number,
            page_type=KIOSK_PAGE_TYPE,
            observation_period=observation_period,
            comparison_period=None,
            entity_type=KIOSK_ENTITY_TYPE,
            entity_name=KIOSK_ENTITY_NAME,
            indicator="kiosk_count",
            value=values.observed_count,
            unit="count",
            source_bbox=calculate_bbox([observed_word]),
        ),
        StructuredMeasure(
            document_id=document_id,
            file_name=file_name,
            page_number=page_number,
            page_type=KIOSK_PAGE_TYPE,
            observation_period=observation_period,
            comparison_period=comparison_period,
            entity_type=KIOSK_ENTITY_TYPE,
            entity_name=KIOSK_ENTITY_NAME,
            indicator="kiosk_count_change",
            value=values.absolute_change,
            unit="count",
            source_bbox=calculate_bbox([absolute_change_word]),
        ),
        StructuredMeasure(
            document_id=document_id,
            file_name=file_name,
            page_number=page_number,
            page_type=KIOSK_PAGE_TYPE,
            observation_period=observation_period,
            comparison_period=comparison_period,
            entity_type=KIOSK_ENTITY_TYPE,
            entity_name=KIOSK_ENTITY_NAME,
            indicator="kiosk_count_change_pct",
            value=values.percentage_change,
            unit="percent",
            source_bbox=calculate_bbox([percentage_change_word]),
        ),
    ]

    if values.revenue_change_percentage is not None:
        revenue_change_word = numeric_words[4]

        measures.append(
            StructuredMeasure(
                document_id=document_id,
                file_name=file_name,
                page_number=page_number,
                page_type=KIOSK_PAGE_TYPE,
                observation_period=observation_period,
                comparison_period=comparison_period,
                entity_type=KIOSK_ENTITY_TYPE,
                entity_name=KIOSK_ENTITY_NAME,
                indicator="kiosk_revenue_change_pct",
                value=values.revenue_change_percentage,
                unit="percent",
                source_bbox=calculate_bbox([revenue_change_word]),
            )
        )

    return measures

def extract_kiosk_measures(
    pdf_path: str | Path,
) -> list[StructuredMeasure]:
    path = Path(pdf_path)

    document_id = build_document_id(path)
    metadata = extract_report_metadata(path.name)

    report_year = metadata["report_year"]
    report_month = metadata["report_month"]

    if report_year is None or report_month is None:
        raise ValueError(f"Période introuvable dans {path.name!r}.")

    if report_month not in MONTH_NUMBERS:
        raise ValueError(f"Mois non pris en charge : {report_month!r}.")

    month_number = MONTH_NUMBERS[report_month]

    observation_period = f"{report_year:04d}-{month_number:02d}"
    comparison_period = f"{report_year - 1:04d}-{month_number:02d}"

    with pymupdf.open(path) as document:
        page = document[3]  # page 4 du rapport
        words = page.get_text("words")

        numeric_words = extract_numeric_row_tokens(
            words=words,
            expected_label=KIOSK_ENTITY_NAME,
        )

        values = parse_numeric_row_values(numeric_words)

        measures = build_kiosk_measures(
            values=values,
            numeric_words=numeric_words,
            document_id=document_id,
            file_name=path.name,
            page_number=4,
            observation_period=observation_period,
            comparison_period=comparison_period,
        )

    return measures

OUTLET_DISTRIBUTION_TITLE = "Répartition et évolution par type d'enseigne"

OUTLET_DISTRIBUTION_PAGE_TYPE = "outlet_type_distribution"

MONTH_NUMBERS = {
    "juin": 6,
    "decembre": 12,
}
OUTLET_DISTRIBUTION_TITLE = "Répartition et évolution par type d'enseigne"

OUTLET_DISTRIBUTION_PAGE_TYPE = "outlet_type_distribution"

MONTH_NUMBERS = {
    "juin": 6,
    "decembre": 12,
}

OUTLET_DISTRIBUTION_COLUMNS = [
    ColumnSpec(
        zone=(180, 250),
        indicator="outlet_count",
        unit="count",
        parser=parse_integer,
        is_comparison=False,
    ),
    ColumnSpec(
        zone=(250, 330),
        indicator="outlet_count_change",
        unit="count",
        parser=parse_integer,
        is_comparison=True,
    ),
    ColumnSpec(
        zone=(330, 415),
        indicator="outlet_count_change_pct",
        unit="percent",
        parser=parse_percentage,
        is_comparison=True,
    ),
    ColumnSpec(
        zone=(415, 495),
        indicator="linear_km",
        unit="km",
        parser=parse_decimal,
        is_comparison=False,
    ),
    ColumnSpec(
        zone=(495, 570),
        indicator="linear_km_change",
        unit="km",
        parser=parse_decimal,
        is_comparison=True,
    ),
    ColumnSpec(
        zone=(570, 675),
        indicator="linear_km_change_pct",
        unit="percent",
        parser=parse_percentage,
        is_comparison=True,
    ),
    ColumnSpec(
        zone=(675, 790),
        indicator="press_revenue_change_pct",
        unit="percent",
        parser=parse_percentage,
        is_comparison=True,
    ),
]

def extract_numeric_row_tokens(
    words: list[PdfWord],
    expected_label: str,
    row_tolerance: float = 4.0,
) -> list[PdfWord]:
    normalized_expected_label = normalize_text(expected_label)

    label_word = next(
        (
            word
            for word in words
            if normalize_text(str(word[4])) == normalized_expected_label
        ),
        None,
    )

    if label_word is None:
        raise ValueError(f"Libellé introuvable : {expected_label!r}")

    target_y = float(label_word[1])

    numeric_words = [
        word
        for word in words
        if (
            abs(float(word[1]) - target_y) <= row_tolerance
            and is_numeric_token(str(word[4]))
        )
    ]

    numeric_words.sort(key=lambda word: float(word[0]))

    return numeric_words

def parse_numeric_row_values(
    numeric_words: list[PdfWord],
) -> NumericRowValues:
    if len(numeric_words) not in (4, 5):
        raise ValueError(
            f"Nombre inattendu de valeurs numériques : {len(numeric_words)}"
        )

    values = [str(word[4]) for word in numeric_words]

    comparison_count = parse_integer(values[0])
    observed_count = parse_integer(values[1])
    absolute_change = parse_integer(values[2])
    percentage_change = parse_percentage(values[3])

    revenue_change_percentage = (
        parse_percentage(values[4])
        if len(values) == 5
        else None
    )

    return NumericRowValues(
        comparison_count=comparison_count,
        observed_count=observed_count,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        revenue_change_percentage=revenue_change_percentage,
    )


def extract_outlet_distribution_measures(
    pdf_path: str | Path,
    entity_names: Sequence[str] = OUTLET_TYPE_ENTITIES,
    row_tolerance: float = 4.0,
) -> list[StructuredMeasure]:
    path = Path(pdf_path)

    document_id = build_document_id(path)

    metadata = extract_report_metadata(path.name)

    report_year = metadata["report_year"]
    report_month = metadata["report_month"]

    if report_year is None or report_month is None:
        raise ValueError(f"Période introuvable dans {path.name!r}.")

    if report_month not in MONTH_NUMBERS:
        raise ValueError(f"Mois non pris en charge : {report_month!r}.")

    month_number = MONTH_NUMBERS[report_month]

    observation_period = f"{report_year:04d}-{month_number:02d}"

    comparison_period = f"{report_year - 1:04d}-{month_number:02d}"

    measures: list[StructuredMeasure] = []

    with pymupdf.open(path) as document:
        page_index = find_page_by_title(
            document=document,
            expected_title=OUTLET_DISTRIBUTION_TITLE,
        )

        page = document[page_index]
        words = page.get_text("words")

        for entity_name in entity_names:
            row_y = find_row_y(
                words=words,
                expected_label=entity_name,
            )

            for spec in OUTLET_DISTRIBUTION_COLUMNS:
                raw_value, bbox = extract_cell(
                    words=words,
                    zone=spec.zone,
                    target_y=row_y,
                    tolerance=row_tolerance,
                )

                typed_value = spec.parser(raw_value)

                measure = StructuredMeasure(
                    document_id=document_id,
                    file_name=path.name,
                    page_number=page_index + 1,
                    page_type=OUTLET_DISTRIBUTION_PAGE_TYPE,
                    observation_period=observation_period,
                    comparison_period=(
                        comparison_period
                        if spec.is_comparison
                        else None
                    ),
                    entity_type="outlet_type",
                    entity_name=entity_name,
                    indicator=spec.indicator,
                    value=typed_value,
                    unit=spec.unit,
                    source_bbox=bbox,
                )

                measures.append(measure)

    return measures


def extract_outlet_type_measures(
    pdf_path: str | Path,
    entity_name: str,
    row_tolerance: float = 4.0,
) -> list[StructuredMeasure]:
    """
    Interface pratique pour extraire une seule catégorie.
    """
    return extract_outlet_distribution_measures(
        pdf_path=pdf_path,
        entity_names=[entity_name],
        row_tolerance=row_tolerance,
    )


ADDITIVE_ENTITIES = (
    "RESEAU TRADITIONNEL",
    "ENSEIGNES PRESSE",
    "ENSEIGNES NON PRESSE",
    "AUTRES",
)

ADDITIVE_INDICATORS = (
    "outlet_count",
    "outlet_count_change",
    "linear_km",
    "linear_km_change",
)


def validate_additive_totals(
    measures: list[StructuredMeasure],
    absolute_tolerance: float = 0.11,
) -> list[str]:
    """
    Vérifie que la somme des catégories correspond
    au total affiché pour les indicateurs additifs.
    """
    issues: list[str] = []

    values = {
        (
            measure.entity_name,
            measure.indicator,
        ): float(measure.value)
        for measure in measures
    }

    for indicator in ADDITIVE_INDICATORS:
        category_keys = [
            (entity_name, indicator)
            for entity_name in ADDITIVE_ENTITIES
        ]

        total_key = (
            "TOTAL",
            indicator,
        )

        missing_keys = [
            key
            for key in [*category_keys, total_key]
            if key not in values
        ]

        if missing_keys:
            issues.append(
                f"{indicator}: mesures absentes {missing_keys}"
            )
            continue

        calculated_total = sum(
            values[key]
            for key in category_keys
        )

        displayed_total = values[total_key]

        if not math.isclose(
            calculated_total,
            displayed_total,
            abs_tol=absolute_tolerance,
        ):
            issues.append(
                f"{indicator}: "
                f"somme={calculated_total:g}, "
                f"total affiché={displayed_total:g}"
            )

    return issues


def audit_additive_totals(
    measures: list[StructuredMeasure],
    exact_tolerance: float = 0.11,
    rounding_tolerance: float = 1.01,
) -> ValidationReport:
    """
    Classe les écarts entre erreurs et avertissements.

    Un petit écart peut provenir de l'addition
    de valeurs individuellement arrondies.
    """
    errors: list[str] = []
    warnings: list[str] = []

    values = {
        (
            measure.entity_name,
            measure.indicator,
        ): float(measure.value)
        for measure in measures
    }

    for indicator in ADDITIVE_INDICATORS:
        category_keys = [
            (entity_name, indicator)
            for entity_name in ADDITIVE_ENTITIES
        ]

        total_key = (
            "TOTAL",
            indicator,
        )

        required_keys = [
            *category_keys,
            total_key,
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in values
        ]

        if missing_keys:
            errors.append(
                f"{indicator}: mesures absentes {missing_keys}"
            )
            continue

        calculated_total = sum(
            values[key]
            for key in category_keys
        )

        displayed_total = values[total_key]

        difference = abs(
            calculated_total - displayed_total
        )

        if difference <= exact_tolerance:
            continue

        message = (
            f"{indicator}: "
            f"somme={calculated_total:g}, "
            f"total affiché={displayed_total:g}, "
            f"écart={difference:g}"
        )

        if difference <= rounding_tolerance:
            warnings.append(
                f"{message} (écart d'arrondi toléré)"
            )
        else:
            errors.append(message)

    return ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
    )