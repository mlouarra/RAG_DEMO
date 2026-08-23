from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_structured_payload(
    json_path: str | Path,
) -> dict[str, Any]:
    """
    Charge et valide la structure générale du fichier
    de mesures structurées.
    """
    path = Path(json_path)

    if not path.is_file():
        raise FileNotFoundError(f"Fichier structuré introuvable : {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Le fichier JSON est invalide : {path}") from error

    if not isinstance(payload, dict):
        raise ValueError("La racine du fichier JSON doit être un objet.")

    required_fields = {
        "schema_version",
        "document_count",
        "measure_count",
        "measures",
    }

    missing_fields = required_fields - payload.keys()

    if missing_fields:
        missing_fields_text = ", ".join(sorted(missing_fields))

        raise ValueError(f"Champs obligatoires manquants : {missing_fields_text}")

    if not isinstance(payload["measures"], list):
        raise ValueError("Le champ 'measures' doit être une liste.")

    actual_measure_count = len(payload["measures"])

    if payload["measure_count"] != actual_measure_count:
        raise ValueError(
            "Nombre de mesures incohérent : "
            f"{payload['measure_count']} annoncé, "
            f"{actual_measure_count} trouvé."
        )

    return payload


def validate_period(period: str) -> str:
    """
    Valide une période au format ISO YYYY-MM.
    """
    try:
        parsed_period = datetime.strptime(
            period,
            "%Y-%m",
        )
    except ValueError as error:
        raise ValueError(f"Période invalide : {period!r}. Format attendu : YYYY-MM.") from error

    return parsed_period.strftime("%Y-%m")


def select_measure_series(
    payload: dict[str, Any],
    entity_name: str,
    indicator: str,
    start_period: str | None = None,
    end_period: str | None = None,
) -> list[dict[str, Any]]:
    """
    Sélectionne et trie chronologiquement les mesures
    correspondant à une entité et à un indicateur.
    """
    normalized_entity_name = entity_name.strip().upper()
    normalized_indicator = indicator.strip()

    if start_period is not None:
        start_period = validate_period(start_period)

    if end_period is not None:
        end_period = validate_period(end_period)

    if start_period is not None and end_period is not None and start_period > end_period:
        raise ValueError("La période de début doit être antérieure ou égale à la période de fin.")

    selected_measures = [
        measure
        for measure in payload["measures"]
        if (
            str(measure.get("entity_name", "")).strip().upper() == normalized_entity_name
            and measure.get("indicator") == normalized_indicator
            and (start_period is None or measure["observation_period"] >= start_period)
            and (end_period is None or measure["observation_period"] <= end_period)
        )
    ]

    selected_measures.sort(key=lambda measure: measure["observation_period"])

    return selected_measures


def calculate_series_evolution(
    measures: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calcule l'évolution entre la première et la dernière
    mesure d'une série chronologique.
    """
    if len(measures) < 2:
        raise ValueError("Au moins deux mesures sont nécessaires pour calculer une évolution.")

    first_measure = measures[0]
    last_measure = measures[-1]

    first_value = first_measure["value"]
    last_value = last_measure["value"]

    if (
        not isinstance(first_value, (int, float))
        or isinstance(first_value, bool)
        or not isinstance(last_value, (int, float))
        or isinstance(last_value, bool)
    ):
        raise ValueError("Les valeurs de la série doivent être numériques.")

    absolute_change = last_value - first_value

    if first_value == 0:
        percentage_change = None
    else:
        percentage_change = (absolute_change / first_value) * 100

    return {
        "entity_name": first_measure["entity_name"],
        "indicator": first_measure["indicator"],
        "unit": first_measure["unit"],
        "start_period": first_measure["observation_period"],
        "start_value": first_value,
        "end_period": last_measure["observation_period"],
        "end_value": last_value,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "start_source": {
            "file_name": first_measure["file_name"],
            "page_number": first_measure["page_number"],
            "source_bbox": first_measure["source_bbox"],
        },
        "end_source": {
            "file_name": last_measure["file_name"],
            "page_number": last_measure["page_number"],
            "source_bbox": last_measure["source_bbox"],
        },
    }

def build_long_term_evolution_facts(
    payload: dict[str, Any],
    entities: list[str],
    indicator: str,
) -> list[dict[str, Any]]:
    """
    Construit des faits candidats décrivant l'évolution
    de long terme d'un indicateur pour plusieurs entités.

    Cette fonction ne décide pas si un fait est marquant.
    Elle s'appuie sur les séries structurées et sur les
    calculs réalisés par calculate_series_evolution().
    """
    facts: list[dict[str, Any]] = []

    for entity_name in entities:
        measures = select_measure_series(
            payload=payload,
            entity_name=entity_name,
            indicator=indicator,
        )

        if len(measures) < 2:
            continue

        evolution = calculate_series_evolution(measures)

        facts.append(
            {
                "fact_type": "long_term_evolution",
                "entity_name": evolution["entity_name"],
                "indicator": evolution["indicator"],
                "unit": evolution["unit"],
                "start_period": evolution["start_period"],
                "start_value": evolution["start_value"],
                "end_period": evolution["end_period"],
                "end_value": evolution["end_value"],
                "absolute_change": evolution["absolute_change"],
                "percentage_change": evolution["percentage_change"],
                "start_source": evolution["start_source"],
                "end_source": evolution["end_source"],
            }
        )

    return facts

def build_period_change_facts(
    payload: dict[str, Any],
    entities: list[str],
    indicator: str,
) -> list[dict[str, Any]]:
    """
    Construit des faits candidats à partir des variations
    publiées pour chaque période.

    La fonction ne décide pas encore si une variation
    est suffisamment importante pour être considérée
    comme un fait marquant.
    """
    facts: list[dict[str, Any]] = []

    for entity_name in entities:
        measures = select_measure_series(
            payload=payload,
            entity_name=entity_name,
            indicator=indicator,
        )

        for measure in measures:
            facts.append(
                {
                    "fact_type": "period_change",
                    "entity_name": measure["entity_name"],
                    "indicator": measure["indicator"],
                    "unit": measure["unit"],
                    "observation_period": measure["observation_period"],
                    "comparison_period": measure["comparison_period"],
                    "value": measure["value"],
                    "source": {
                        "file_name": measure["file_name"],
                        "page_number": measure["page_number"],
                        "source_bbox": measure["source_bbox"],
                    },
                }
            )

    return facts
def select_strongest_period_changes(
    facts: list[dict[str, Any]],
    start_period: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Sélectionne les variations périodiques les plus fortes
    en valeur absolue.

    Un filtre optionnel permet d'exclure les faits dont
    la période de comparaison est antérieure au début
    de la période étudiée.
    """
    if limit <= 0:
        raise ValueError("limit doit être strictement positif.")

    selected_facts = facts

    if start_period is not None:
        start_period = validate_period(start_period)

        selected_facts = [
            fact
            for fact in selected_facts
            if (
                fact["comparison_period"] is not None
                and fact["comparison_period"] >= start_period
            )
        ]

    selected_facts = sorted(
        selected_facts,
        key=lambda fact: abs(float(fact["value"])),
        reverse=True,
    )

    return selected_facts[:limit]

def query_measure_evolution(
    payload: dict[str, Any],
    entity_name: str,
    indicator: str,
    start_period: str | None = None,
    end_period: str | None = None,
) -> dict[str, Any]:
    """
    Exécute une requête d'évolution sur les données
    structurées et retourne les mesures avec le calcul.
    """
    measures = select_measure_series(
        payload=payload,
        entity_name=entity_name,
        indicator=indicator,
        start_period=start_period,
        end_period=end_period,
    )

    if not measures:
        raise ValueError("Aucune mesure trouvée pour les critères demandés.")

    if len(measures) < 2:
        raise ValueError(
            "Une seule mesure a été trouvée. "
            "Le calcul d'une évolution nécessite "
            "au moins deux observations."
        )

    evolution = calculate_series_evolution(measures)

    return {
        "query": {
            "entity_name": entity_name,
            "indicator": indicator,
            "start_period": start_period,
            "end_period": end_period,
        },
        "measure_count": len(measures),
        "measures": measures,
        "evolution": evolution,
    }
