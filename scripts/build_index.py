from __future__ import annotations

import json

from fm_rag.config import load_config
from fm_rag.embeddings import OllamaEmbeddingClient
from fm_rag.indexer import build_index


def main() -> None:
    config = load_config()

    paths_config = config["paths"]
    ollama_config = config["ollama"]

    embedding_client = OllamaEmbeddingClient(
        base_url=ollama_config["base_url"],
        model=ollama_config["embedding_model"],
        timeout=ollama_config["timeout_seconds"],
    )

    manifest = build_index(
        documents_dir=paths_config["documents_dir"],
        index_dir=paths_config["index_dir"],
        embedding_client=embedding_client,
        batch_size=8,
    )

    print()
    print("Index construit avec succès :")
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
