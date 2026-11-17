import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.bm25 import BM25Searcher
from app.dense import DenseSearcher
from app.documents import load_documents
from app.hybrid import HybridSearcher
from app.schemas import SearchInput, SearchMode, SearchOutput, SearchResult
from app.searcher import Searcher

searchers: dict[SearchMode, Searcher] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    documents = load_documents()
    bm25 = BM25Searcher(documents)
    dense = DenseSearcher(documents)
    searchers[SearchMode.bm25] = bm25
    searchers[SearchMode.dense] = dense
    searchers[SearchMode.hybrid] = HybridSearcher([bm25, dense])
    yield
    searchers.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def get_health():
    return {"status": "ok"}


@app.post("/search")
def search(body: SearchInput) -> SearchOutput:
    start = time.perf_counter()
    hits = searchers[body.mode].search(body.query, body.maxOutput)
    took_ms = int((time.perf_counter() - start) * 1000)

    return SearchOutput(
        query=body.query,
        mode=body.mode,
        found=bool(hits),
        results=[SearchResult(id=h.document.id, text=h.document.text, score=h.score) for h in hits],
        took_ms=took_ms,
    )
