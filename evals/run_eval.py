"""Оценка качества поиска на RuBQ: сравнение bm25, dense и hybrid.

Запуск из корня проекта:
    uv run python -m evals.run_eval
"""
import json
import os
import statistics
import time
from datetime import datetime

from datasets import load_dataset

from app.bm25 import BM25Searcher
from app.config import ROOT
from app.dense import DenseSearcher
from app.documents import load_documents
from app.hybrid import HybridSearcher
from app.searcher import Searcher
from evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DATASET = "mteb/RuBQRetrieval"
RESULTS_DIR = ROOT / "evals" / "results"
K = 10


def load_queries_and_qrels() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Вопросы {query_id: текст} и правильные ответы {query_id: {doc_id: оценка}}."""
    token = os.getenv("HF_TOKEN")
    # У датасета три конфига: corpus, queries, qrels. Во всех один split — "test".
    queries_ds = load_dataset(DATASET, "queries", token=token)["test"]
    qrels_ds = load_dataset(DATASET, "qrels", token=token)["test"]

    queries = {row["_id"]: row["text"] for row in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        if row["score"] > 0:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
    return queries, qrels


def evaluate(searcher: Searcher, queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict:
    recalls, mrrs, ndcgs, latencies = [], [], [], []
    for query_id, relevant in qrels.items():
        start = time.perf_counter()
        hits = searcher.search(queries[query_id], K)
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved = [hit.document.id for hit in hits]
        recalls.append(recall_at_k(retrieved, relevant, K))
        mrrs.append(mrr_at_k(retrieved, relevant, K))
        ndcgs.append(ndcg_at_k(retrieved, relevant, K))

    return {
        f"recall@{K}": statistics.mean(recalls),
        f"mrr@{K}": statistics.mean(mrrs),
        f"ndcg@{K}": statistics.mean(ndcgs),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": statistics.quantiles(latencies, n=20)[-1],
        # Доля вопросов, где не нашли ни одного правильного документа
        "zero_hit_rate": sum(1 for r in recalls if r == 0) / len(recalls),
    }


def main() -> None:
    queries, qrels = load_queries_and_qrels()
    documents = load_documents()
    corpus_ids = {doc.id for doc in documents}
    print(f"Документов: {len(documents)}, вопросов: {len(queries)}, вопросов с ответами: {len(qrels)}")

    # Проверки: всё ли из qrels есть в корпусе и среди вопросов
    missing_pairs = [(q, d) for q, rel in qrels.items() for d in rel if d not in corpus_ids]
    queries_with_missing = {q for q, _ in missing_pairs}
    no_text = [q for q in qrels if q not in queries]
    print(f"Пар (вопрос, документ), где документа нет в корпусе: {len(missing_pairs)} "
          f"(затронуто вопросов: {len(queries_with_missing)})")
    print(f"Вопросов из qrels без текста: {len(no_text)}")
    qrels = {q: rel for q, rel in qrels.items() if q in queries}

    # Searcher-ы создаём один раз, как в lifespan в app/main.py
    t = time.perf_counter()
    bm25 = BM25Searcher(documents)
    dense = DenseSearcher(documents)
    searchers: dict[str, Searcher] = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": HybridSearcher([bm25, dense]),
    }
    print(f"Индексы построены за {time.perf_counter() - t:.1f} с")

    # Прогрев: первый запрос к dense-модели медленный, не хотим, чтобы он портил задержку
    dense.search("прогрев", K)

    results = {}
    for mode, searcher in searchers.items():
        t = time.perf_counter()
        results[mode] = evaluate(searcher, queries, qrels)
        print(f"{mode}: готово за {time.perf_counter() - t:.1f} с")

    print()
    header = f"{'режим':<8} {'Recall@10':>10} {'MRR@10':>8} {'nDCG@10':>8} {'0 найдено':>10} {'ms mean':>8} {'ms p50':>7} {'ms p95':>7}"
    print(header)
    print("-" * len(header))
    for mode, m in results.items():
        print(f"{mode:<8} {m['recall@10']:>10.4f} {m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f} "
              f"{m['zero_hit_rate']:>10.1%} {m['latency_ms_mean']:>8.1f} {m['latency_ms_p50']:>7.1f} "
              f"{m['latency_ms_p95']:>7.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"eval_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    report = {
        "dataset": DATASET,
        "k": K,
        "n_documents": len(documents),
        "n_queries": len(qrels),
        "missing_relevant_pairs": len(missing_pairs),
        "queries_with_missing_relevant": len(queries_with_missing),
        "results": results,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
