"""Расчёт эмбеддингов документов и сохранение в embeddings.npy."""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, EMBEDDINGS_PATH
from app.documents import load_documents


def main() -> None:
    if EMBEDDINGS_PATH.exists():
        print(f"{EMBEDDINGS_PATH} уже есть, пропускаю")
        return

    documents = load_documents()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # e5 обучена с префиксами: "passage: " для документов, "query: " для запросов
    embeddings = model.encode(
        [f"passage: {doc.text}" for doc in documents],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    print(f"Сохранено {embeddings.shape} в {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
