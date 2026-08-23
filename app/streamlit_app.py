from pathlib import Path

import streamlit as st

from fm_rag.rag_service import (
    answer_kiosk_evolution,
    answer_network_highlights,
)


PRESS_CHART_PATH = Path(
    "data/structured/enseignes_presse_outlet_count.png"
)


st.set_page_config(
    page_title="France Messagerie - Démonstrateur RAG",
    page_icon="📰",
    layout="centered",
)


st.title("France Messagerie")
st.subheader("Démonstrateur d'analyse documentaire")

st.write(
    "Analyse des rapports de suivi du réseau "
    "de décembre 2021 à juin 2026."
)

st.divider()

use_case = st.selectbox(
    "Choisissez une démonstration",
    [
        "Évolution des kiosques",
        "Faits marquants du réseau",
        "Graphique comparatif Enseignes Presse",
    ],
)

run_button = st.button("Analyser")

st.divider()

st.subheader("Résultat")

if run_button:

    if use_case == "Évolution des kiosques":

        question = (
            "Fais-moi une synthèse de l'évolution des kiosques "
            "depuis décembre 2021"
        )

        with st.spinner("Analyse en cours..."):
            answer = answer_kiosk_evolution(question)

        st.write(answer)

    elif use_case == "Faits marquants du réseau":

        question = (
            "Identifie-moi 5 à 10 faits marquants sur le réseau "
            "des marchands de journaux depuis décembre 2021 "
            "jusqu'à juin 2026"
        )

        with st.spinner("Analyse en cours..."):
            answer = answer_network_highlights(question)

        st.write(answer)

    elif use_case == "Graphique comparatif Enseignes Presse":

        if PRESS_CHART_PATH.exists():
            st.image(
                str(PRESS_CHART_PATH),
                caption="Évolution du nombre de points de vente - Enseignes Presse",
                use_container_width=True,
            )
        else:
            st.error(
                "Le graphique Enseignes Presse n'a pas encore été généré."
            )