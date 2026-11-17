"""Настройки приложения."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS_PATH = ROOT / "data" / "docs.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
