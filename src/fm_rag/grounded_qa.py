from __future__ import annotations

from collections.abc import Sequence

from fm_rag.schemas import RetrievedChunk


def extract_relevant_excerpt(
    text: str,
    keyword: str,
    context_size: int = 250,
) -> str:
    """
    Extrait une portion de texte autour de la première
    occurrence d'un mot-clé.

    Si le mot-clé n'est pas trouvé, retourne le texte
    complet.
    """
    normalized_text = text.strip()
    normalized_keyword = keyword.strip()

    if not normalized_text:
        return ""

    if not normalized_keyword:
        raise ValueError("Le mot-clé ne doit pas être vide.")

    if context_size < 0:
        raise ValueError("context_size doit être positif ou nul.")

    position = normalized_text.casefold().find(
        normalized_keyword.casefold()
    )

    if position == -1:
        return normalized_text

    start = max(
        0,
        position - context_size,
    )

    end = min(
        len(normalized_text),
        position + len(normalized_keyword) + context_size,
    )

    return normalized_text[start:end].strip()


def build_grounded_messages(
    question: str,
    structured_context: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    keyword: str | None = None,
    include_rag_text: bool = True,
) -> list[dict[str, str]]:
    """
    Construit les messages envoyés au LLM local.

    Les données structurées calculées par Python sont
    considérées comme la source de vérité pour les valeurs
    numériques et les évolutions.

    Le contexte RAG apporte uniquement un contexte
    documentaire complémentaire et les références aux sources.

    Paramètres :
    - keyword :
        si renseigné, le texte RAG est réduit à un extrait
        centré autour de ce mot-clé ;
    - include_rag_text :
        si False, le contenu textuel brut des chunks n'est
        pas transmis au LLM. Seules les références
        documentaires sont conservées.
    """
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("La question ne doit pas être vide.")

    rag_sections: list[str] = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.metadata

        file_name = metadata.get(
            "file_name",
            "document inconnu",
        )

        page_number = metadata.get(
            "page_number",
            "?",
        )

        if not include_rag_text:
            excerpt = (
                "Source documentaire utilisée pour "
                "contextualiser l'analyse."
            )

        elif keyword is None:
            excerpt = chunk.text.strip()

        else:
            excerpt = extract_relevant_excerpt(
                text=chunk.text,
                keyword=keyword,
                context_size=250,
            )

        rag_sections.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Document : {file_name}",
                    f"Page : {page_number}",
                    excerpt,
                ]
            )
        )

    if rag_sections:
        rag_context = "\n\n".join(rag_sections)
    else:
        rag_context = (
            "Aucun contexte documentaire complémentaire."
        )

    system_message = """
Tu es un assistant d'analyse documentaire pour France Messagerie.

Tu dois répondre uniquement à partir des informations fournies.

Règles impératives :
- les DONNÉES STRUCTURÉES VALIDÉES PAR PYTHON constituent la source de vérité pour toutes les valeurs numériques ;
- utilise exclusivement ces données structurées pour décrire les niveaux, minimums, maximums et évolutions ;
- ne recalcule pas les valeurs déjà fournies ;
- ne calcule pas de variation numérique si elle n'est pas explicitement fournie dans les DONNÉES STRUCTURÉES VALIDÉES PAR PYTHON ;
- lorsque seules les valeurs d'une série sont fournies, présente les valeurs sans ajouter toi-même de pourcentages ou de différences entre périodes ;
- ne compare pas entre eux les nombres présents dans le contexte documentaire RAG ;
- les nombres présents dans les extraits RAG peuvent correspondre à des périodes de comparaison différentes ;
- utilise le contexte documentaire RAG uniquement pour confirmer les documents, les pages, les intitulés et apporter du contexte non contradictoire ;
- en cas de divergence apparente entre le RAG et les données structurées, conserve les données structurées ;
- ne signale pas d'incertitude ou de divergence si les données structurées fournissent directement la réponse ;
- distingue les observations de décembre des observations de juin lorsque cela est pertinent ;
- n'invente aucune donnée absente ;
- réponds uniquement au périmètre de la question posée ;
- rédige une réponse claire, concise et professionnelle en français.
""".strip()

    user_message = f"""
QUESTION

{normalized_question}


DONNÉES STRUCTURÉES VALIDÉES PAR PYTHON

{structured_context}


CONTEXTE DOCUMENTAIRE RETROUVÉ PAR LE RAG

{rag_context}


Rédige une réponse synthétique à la question.

Pour les chiffres et les évolutions, utilise uniquement la section
"DONNÉES STRUCTURÉES VALIDÉES PAR PYTHON".

Le contexte RAG sert uniquement de justification documentaire.

Ne calcule pas toi-même des variations numériques qui ne sont pas
explicitement fournies par Python.

Réponds uniquement au périmètre de la question posée.
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]