from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import requests


ChatMessage = Mapping[str, str]
ResponseFormat = str | dict[str, Any]


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def chat(
        self,
        messages: Sequence[ChatMessage],
        response_format: ResponseFormat | None = None,
    ) -> str:
        """
        Envoie une conversation à Ollama et retourne
        uniquement le contenu textuel de la réponse.
        """
        normalized_messages = [
            {
                "role": message["role"].strip(),
                "content": message["content"].strip(),
            }
            for message in messages
        ]

        if not normalized_messages:
            raise ValueError(
                "Au moins un message est nécessaire."
            )

        if any(
            not message["role"] or not message["content"]
            for message in normalized_messages
        ):
            raise ValueError(
                "Le rôle et le contenu des messages "
                "ne doivent pas être vides."
            )

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        if response_format is not None:
            request_payload["format"] = response_format

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=request_payload,
            timeout=self.timeout,
        )

        response.raise_for_status()
        response_payload = response.json()

        message = response_payload.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Réponse Ollama invalide : champ "
                "'message' absent."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Réponse Ollama invalide : contenu vide."
            )

        return content.strip()