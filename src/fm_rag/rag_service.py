from __future__ import annotations

from pathlib import Path

from fm_rag.config import load_config
from fm_rag.embeddings import OllamaEmbeddingClient
from fm_rag.grounded_qa import build_grounded_messages
from fm_rag.ollama_client import OllamaChatClient
from fm_rag.retriever import DenseRetriever
from fm_rag.structured_repository import (
    build_long_term_evolution_facts,
    build_period_change_facts,
    calculate_series_evolution,
    load_structured_payload,
    select_measure_series,
    select_strongest_period_changes,
)


KIOSK_DATA_PATH = Path("data/structured/kiosk_evolution.json")
OUTLET_DISTRIBUTION_DATA_PATH = Path(
    "data/structured/outlet_distribution.json"
)


def build_kiosk_structured_context() -> str:
    """
    Construit le contexte numérique validé pour
    l'évolution des kiosques.
    """
    payload = load_structured_payload(KIOSK_DATA_PATH)

    measures = select_measure_series(
        payload=payload,
        entity_name="Kiosques",
        indicator="kiosk_count",
    )

    evolution = calculate_series_evolution(measures)

    series_lines = [
        f"{measure['observation_period']} : {measure['value']} kiosques"
        for measure in measures
    ]

    percentage_change = evolution["percentage_change"]

    if percentage_change is None:
        percentage_text = "non calculable"
    else:
        percentage_text = f"{percentage_change:.2f} %"

    evolution_lines = [
        "",
        "Évolution globale :",
        (
            f"{evolution['start_period']} : "
            f"{evolution['start_value']} kiosques"
        ),
        (
            f"{evolution['end_period']} : "
            f"{evolution['end_value']} kiosques"
        ),
        (
            f"Variation absolue : "
            f"{evolution['absolute_change']:+g} kiosques"
        ),
        f"Variation relative : {percentage_text}",
    ]

    return "\n".join(
        [
            *series_lines,
            *evolution_lines,
        ]
    )


def build_network_highlights_structured_context() -> str:
    """
    Construit un contexte structuré contenant des faits
    candidats sur l'évolution du réseau.

    Le contexte combine :
    - les évolutions de long terme ;
    - les variations périodiques les plus fortes.

    Cette fonction prépare des faits validés par Python.
    Elle ne demande pas au LLM d'effectuer les calculs.
    """
    payload = load_structured_payload(
        OUTLET_DISTRIBUTION_DATA_PATH
    )

    entities = [
        "RESEAU TRADITIONNEL",
        "ENSEIGNES PRESSE",
        "ENSEIGNES NON PRESSE",
        "AUTRES",
        "TOTAL",
    ]

    long_term_facts = build_long_term_evolution_facts(
        payload=payload,
        entities=entities,
        indicator="outlet_count",
    )

    period_change_facts = build_period_change_facts(
        payload=payload,
        entities=entities,
        indicator="outlet_count_change_pct",
    )

    strongest_period_changes = select_strongest_period_changes(
        facts=period_change_facts,
        start_period="2021-12",
        limit=5,
    )

    lines: list[str] = []

    lines.append("EVOLUTIONS DE LONG TERME")
    lines.append("")

    for fact in long_term_facts:
        percentage_change = fact["percentage_change"]

        if percentage_change is None:
            percentage_text = "non calculable"
        else:
            percentage_text = f"{percentage_change:+.2f} %"

        lines.append(
            (
                f"- {fact['entity_name']} : "
                f"{fact['start_value']} en {fact['start_period']} "
                f"→ {fact['end_value']} en {fact['end_period']} ; "
                f"variation absolue "
                f"{fact['absolute_change']:+g} ; "
                f"variation relative {percentage_text}"
            )
        )

    lines.append("")
    lines.append("VARIATIONS PERIODIQUES LES PLUS FORTES")
    lines.append("")

    for fact in strongest_period_changes:
        lines.append(
            (
                f"- {fact['entity_name']} : "
                f"{fact['comparison_period']} "
                f"→ {fact['observation_period']} : "
                f"{fact['value']:+.1f} %"
            )
        )

    return "\n".join(lines)


def answer_kiosk_evolution(
    question: str,
) -> str:
    """
    Répond à une question sur l'évolution des kiosques
    en combinant données structurées, RAG et LLM local.
    """
    config = load_config()

    ollama_config = config["ollama"]
    paths_config = config["paths"]

    embedding_client = OllamaEmbeddingClient(
        base_url=ollama_config["base_url"],
        model=ollama_config["embedding_model"],
        timeout=ollama_config["timeout_seconds"],
    )

    retriever = DenseRetriever.from_index(
        index_dir=paths_config["index_dir"],
        embedding_client=embedding_client,
    )

    retrieved_chunks = retriever.retrieve(
        query=question,
        top_k=6,
        metadata_filter={
            "page_number": 4,
        },
    )

    structured_context = build_kiosk_structured_context()

    messages = build_grounded_messages(
    question=question,
    structured_context=structured_context,
    retrieved_chunks=retrieved_chunks,
    keyword="Kiosques",
    include_rag_text=True,
)
    chat_client = OllamaChatClient(
        base_url=ollama_config["base_url"],
        model=ollama_config["chat_model"],
        timeout=ollama_config["timeout_seconds"],
        temperature=config["generation"]["temperature"],
    )

    return chat_client.chat(messages)


def answer_network_highlights(
    question: str,
) -> str:
    """
    Répond à une question sur les faits marquants du réseau
    en combinant faits candidats calculés par Python,
    contexte documentaire RAG et LLM local.
    """
    config = load_config()

    ollama_config = config["ollama"]
    paths_config = config["paths"]

    embedding_client = OllamaEmbeddingClient(
        base_url=ollama_config["base_url"],
        model=ollama_config["embedding_model"],
        timeout=ollama_config["timeout_seconds"],
    )

    retriever = DenseRetriever.from_index(
        index_dir=paths_config["index_dir"],
        embedding_client=embedding_client,
    )

    retrieved_chunks = retriever.retrieve(
        query=question,
        top_k=6,
        metadata_filter={
            "page_number": 3,
        },
    )

    structured_context = (
        build_network_highlights_structured_context()
    )

    messages = build_grounded_messages(
    question=question,
    structured_context=structured_context,
    retrieved_chunks=retrieved_chunks,
    keyword=None,
    include_rag_text=False,
    )

    chat_client = OllamaChatClient(
        base_url=ollama_config["base_url"],
        model=ollama_config["chat_model"],
        timeout=ollama_config["timeout_seconds"],
        temperature=config["generation"]["temperature"],
    )

    return chat_client.chat(messages)