from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

# Permet de générer l'image sans interface graphique.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

INPUT_PATH = Path("data/structured/outlet_distribution.json")

OUTPUT_PATH = Path("data/structured/enseignes_presse_outlet_count.png")


def format_period(period: str) -> str:
    """
    Transforme une période ISO comme 2021-12
    en libellé lisible comme Déc. 2021.
    """
    year, month = period.split("-")

    month_labels = {
        "06": "Juin",
        "12": "Déc.",
    }

    month_label = month_labels.get(
        month,
        month,
    )

    return f"{month_label} {year}"


def load_outlet_counts() -> list[dict]:
    """
    Charge les effectifs des Enseignes Presse
    depuis le fichier de données structurées.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fichier structuré introuvable : {INPUT_PATH}")

    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    measures = [
        measure
        for measure in payload["measures"]
        if (measure["entity_name"] == "ENSEIGNES PRESSE" and measure["indicator"] == "outlet_count")
    ]

    measures.sort(key=lambda measure: measure["observation_period"])

    if not measures:
        raise ValueError("Aucune mesure trouvée pour ENSEIGNES PRESSE / outlet_count.")

    return measures


def main() -> None:
    measures = load_outlet_counts()

    dates = [
        datetime.strptime(
            measure["observation_period"],
            "%Y-%m",
        )
        for measure in measures
    ]

    values = [int(measure["value"]) for measure in measures]

    labels = [format_period(measure["observation_period"]) for measure in measures]

    first_value = values[0]
    last_value = values[-1]

    absolute_change = last_value - first_value

    percentage_change = absolute_change / first_value * 100

    figure, axis = plt.subplots(figsize=(11, 6))

    # Courbe principale
    axis.plot(
        dates,
        values,
        color="#17365D",
        linewidth=2.5,
        marker="o",
        markersize=7,
    )

    # Valeurs affichées au-dessus des points
    for date, value in zip(
        dates,
        values,
        strict=True,
    ):
        formatted_value = f"{value:,}".replace(",", " ")

        axis.annotate(
            formatted_value,
            xy=(date, value),
            xytext=(0, 10),
            textcoords="offset points",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
            color="#17365D",
        )

    # Titre sur une seule ligne pour éviter
    # le chevauchement avec le sous-titre.
    axis.set_title(
        ("Évolution du nombre de points de vente - Enseignes Presse"),
        loc="left",
        fontsize=16,
        fontweight="bold",
        color="#17365D",
        pad=38,
    )

    # Sous-titre contenant l'évolution globale
    axis.text(
        0,
        1.02,
        (f"De {labels[0]} à {labels[-1]} : {absolute_change:+} PdV ({percentage_change:+.1f} %)"),
        transform=axis.transAxes,
        fontsize=10,
        color="#5D6773",
    )

    axis.set_ylabel("Nombre de points de vente")

    # Les positions sont de vraies dates.
    # L'intervalle déc. 2025 → juin 2026
    # représente donc correctement six mois.
    axis.set_xticks(
        dates,
        labels,
    )

    axis.grid(
        axis="y",
        color="#D9E0E7",
        linewidth=0.8,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # Source et méthode d'extraction
    figure.text(
        0.08,
        0.02,
        ("Source : rapports France Messagerie, page 3. Données extraites du PDF natif."),
        fontsize=8,
        color="#6B7280",
    )

    figure.tight_layout(rect=(0, 0.06, 1, 1))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print(f"Points représentés : {len(values)}")
    print(f"Évolution absolue  : {absolute_change:+} PdV")
    print(f"Évolution relative : {percentage_change:+.1f} %")
    print(f"Graphique produit  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
