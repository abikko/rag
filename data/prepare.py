"""Скачивание датасета и сохранение корпуса в docs.jsonl."""
import json
import os

from datasets import load_dataset

from app.config import DOCS_PATH


def main() -> None:
    if DOCS_PATH.exists():
        print(f"{DOCS_PATH} уже есть, пропускаю")
        return

    corpus = load_dataset("mteb/RuBQRetrieval", "corpus", token=os.getenv("HF_TOKEN"))

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for doc in corpus["test"]:
            row = {"_id": doc["_id"], "text": doc["text"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
