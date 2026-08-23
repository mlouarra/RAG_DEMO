from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fm_rag.chunker import build_chunks
from fm_rag.embeddings import OllamaEmbeddingClient
from fm_rag.pdf_loader import load_all_pdf_pages
from fm_rag.schemas import TextChunk


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Au moins un embedding possède une norme nulle.")

    return embeddings / norms


def embed_chunks(
    chunks: list[TextChunk],
    embedding_client: OllamaEmbeddingClient,
    batch_size: int = 8,
) -> np.ndarray:
    if not chunks:
        raise ValueError("Aucun chunk à encoder.")

    if batch_size <= 0:
        raise ValueError("La taille des lots doit être strictement positive.")

    embedding_batches: list[np.ndarray] = []

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        print(f"Encodage des chunks {start + 1} à {end} sur {len(chunks)}...")

        batch_embeddings = embedding_client.embed_batch([chunk.text for chunk in batch])

        embedding_batches.append(batch_embeddings)

    embeddings = np.vstack(embedding_batches).astype(np.float32)

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Le nombre d'embeddings ne correspond pas au nombre de chunks.")

    return normalize_embeddings(embeddings)


def serialize_chunk(chunk: TextChunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def build_index(
    documents_dir: str | Path,
    index_dir: str | Path,
    embedding_client: OllamaEmbeddingClient,
    batch_size: int = 8,
) -> dict[str, Any]:
    documents_path = Path(documents_dir)
    output_path = Path(index_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages = load_all_pdf_pages(documents_path)
    chunks = build_chunks(pages)

    if not pages:
        raise ValueError(f"Aucune page PDF trouvée dans {documents_path}.")

    if not chunks:
        raise ValueError("Aucun chunk n'a été produit.")

    embeddings = embed_chunks(
        chunks=chunks,
        embedding_client=embedding_client,
        batch_size=batch_size,
    )

    chunks_path = output_path / "chunks.json"
    embeddings_path = output_path / "embeddings.npy"
    manifest_path = output_path / "manifest.json"

    chunks_payload = [serialize_chunk(chunk) for chunk in chunks]

    with chunks_path.open("w", encoding="utf-8") as chunks_file:
        json.dump(
            chunks_payload,
            chunks_file,
            ensure_ascii=False,
            indent=2,
        )

    np.save(embeddings_path, embeddings)

    manifest = {
        "document_count": len({page.document_id for page in pages}),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_model": embedding_client.model,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunking_strategy": "one_chunk_per_page",
        "normalized_embeddings": True,
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )

    return manifest
