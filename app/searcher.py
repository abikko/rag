from abc import abstractmethod, ABC
from dataclasses import dataclass

from app.schemas import Document

@dataclass()
class SearchHit:
    document: Document
    score: float

class Searcher(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[SearchHit]:
        ...