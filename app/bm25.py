"""Поиск по словам (BM25)."""
import bm25s

from app.schemas import Document
from app.searcher import Searcher, SearchHit
from app.text import tokenize


class BM25Searcher(Searcher):
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.model = bm25s.BM25()
        self.model.index([doc.tokens() for doc in documents], show_progress=False)

    def search(self, query: str, k: int) -> list[SearchHit]:
        k = min(k, len(self.documents))
        indices, scores = self.model.retrieve([tokenize(query)], k=k, show_progress=False)

        # Документы без единого совпадающего слова получают 0 — это не находки
        return [
            SearchHit(document=self.documents[i], score=float(score))
            for i, score in zip(indices[0], scores[0])
            if score > 0
        ]
