from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import requests


class OllamaEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        embeddings = self.embed_batch([text])

        if embeddings.shape[0] != 1:
            raise ValueError("Ollama n'a pas retourné exactement un embedding.")

        return embeddings[0]

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        inputs = [text.strip() for text in texts]

        if not inputs:
            raise ValueError("Aucun texte à encoder.")

        if any(not text for text in inputs):
            raise ValueError("Les textes à encoder ne doivent pas être vides.")

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": inputs,
                "truncate": True,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        payload = response.json()

        raw_embeddings = payload.get("embeddings")

        if not raw_embeddings:
            raise ValueError("Aucun embedding retourné par Ollama.")

        embeddings = np.asarray(
            raw_embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(f"Forme inattendue des embeddings : {embeddings.shape}")

        if embeddings.shape[0] != len(inputs):
            raise ValueError("Le nombre d'embeddings ne correspond pas au nombre de textes.")

        return embeddings
