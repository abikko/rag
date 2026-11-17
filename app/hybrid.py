"""Гибридный поиск: Reciprocal Rank Fusion (RRF)."""
from app.schemas import Document
from app.searcher import Searcher, SearchHit


class HybridSearcher(Searcher):
    def __init__(self, searchers: list[Searcher], rrf_k: int = 60, candidates: int = 50):
        self.searchers = searchers
        # rrf_k сглаживает разницу между соседними местами: чем больше, тем ровнее
        self.rrf_k = rrf_k
        # Сколько результатов берём у каждого поиска перед объединением
        self.candidates = candidates

    def search(self, query: str, k: int) -> list[SearchHit]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        for searcher in self.searchers:
            hits = searcher.search(query, max(k, self.candidates))
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit.document.id
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (self.rrf_k + rank)
                documents[doc_id] = hit.document

        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [SearchHit(document=documents[doc_id], score=scores[doc_id]) for doc_id in best]
