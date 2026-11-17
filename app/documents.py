"""Загрузка корпуса документов."""
from pathlib import Path

from app.config import DOCS_PATH
from app.schemas import Document


def load_documents(path: Path = DOCS_PATH) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        return [Document.model_validate_json(line) for line in f]
