"""Поиск по смыслу: эмбеддинги в numpy."""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, EMBEDDINGS_PATH
from app.schemas import Document
from app.searcher import Searcher, SearchHit


class DenseSearcher(Searcher):
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = np.load(EMBEDDINGS_PATH)

        if len(self.embeddings) != len(documents):
            raise ValueError(
                f"Эмбеддингов {len(self.embeddings)}, а документов {len(documents)}: "
                "пересчитай data/build_index.py"
            )

    def search(self, query: str, k: int) -> list[SearchHit]:
        k = min(k, len(self.documents))
        query_vector = self.model.encode(f"query: {query}", normalize_embeddings=True)

        # Векторы нормализованы, поэтому скалярное произведение = косинусная близость
        scores = self.embeddings @ query_vector

        # argpartition находит k лучших без полной сортировки всех 56826 оценок
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [SearchHit(document=self.documents[i], score=float(scores[i])) for i in top]
