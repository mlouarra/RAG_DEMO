from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fm_rag.embeddings import OllamaEmbeddingClient
from fm_rag.schemas import RetrievedChunk


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        embedding_client: OllamaEmbeddingClient,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Matrice d'embeddings invalide : {embeddings.shape}")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Le nombre de chunks ne correspond pas au nombre d'embeddings.")

        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_client = embedding_client

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        embedding_client: OllamaEmbeddingClient,
    ) -> DenseRetriever:
        index_path = Path(index_dir)
        chunks_path = index_path / "chunks.json"
        embeddings_path = index_path / "embeddings.npy"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Fichier des chunks introuvable : {chunks_path}")

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Fichier des embeddings introuvable : {embeddings_path}")

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as chunks_file:
            chunks = json.load(chunks_file)

        if not isinstance(chunks, list):
            raise ValueError("Le fichier chunks.json doit contenir une liste.")

        embeddings = np.load(embeddings_path)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            embedding_client=embedding_client,
        )

    @staticmethod
    def normalize_query_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("L'embedding de la question possède une norme nulle.")

        return embedding / norm

    @staticmethod
    def matches_filter(
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        if not metadata_filter:
            return True

        return all(
            metadata.get(key) == expected_value for key, expected_value in metadata_filter.items()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        question = query.strip()

        if not question:
            raise ValueError("La question ne doit pas être vide.")

        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")

        query_embedding = self.embedding_client.embed(question)
        query_embedding = self.normalize_query_embedding(query_embedding)

        scores = self.embeddings @ query_embedding

        candidate_indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if self.matches_filter(
                chunk.get("metadata", {}),
                metadata_filter,
            )
        ]

        ranked_indexes = sorted(
            candidate_indexes,
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results: list[RetrievedChunk] = []

        for index in ranked_indexes[:top_k]:
            chunk = self.chunks[index]

            results.append(
                RetrievedChunk(
                    document_id=chunk["document_id"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(scores[index]),
                    metadata=chunk.get("metadata", {}),
                )
            )

        return results
