"""Метрики качества поиска для одного запроса.

Везде на входе:
- retrieved: id найденных документов по порядку (первый — самый релевантный по мнению поиска);
- relevant: словарь {id: оценка релевантности}. В RuBQ все оценки равны 1,
  но nDCG умеет работать и с градуированными оценками (0, 1, 2, ...).
"""
import math


def recall_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Какую долю правильных документов мы нашли в топ-k."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved[:k] if relevant.get(doc_id, 0) > 0)
    return found / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """1 / (место первого правильного документа). Если в топ-k его нет — 0."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1 / rank
    return 0.0


def dcg_at_k(gains: list[int], k: int) -> float:
    """Сумма оценок со скидкой за место: оценка на месте i делится на log2(i + 1)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains[:k], start=1))


def ndcg_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """DCG нашей выдачи, делённый на DCG идеальной выдачи (0..1)."""
    gains = [relevant.get(doc_id, 0) for doc_id in retrieved]
    ideal = sorted(relevant.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg
