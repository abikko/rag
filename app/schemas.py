from enum import StrEnum

from pydantic import BaseModel, Field

from app.text import tokenize

class SearchMode(StrEnum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"


class SearchInput(BaseModel):
    query: str = Field(min_length=1)
    mode: SearchMode = Field(default=SearchMode.bm25)
    maxOutput: int = Field(default=5, ge=1, le=50)

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class SearchOutput(BaseModel):
    query: str
    mode: SearchMode
    found: bool
    results: list[SearchResult]
    took_ms: int

class Document(BaseModel):
    text: str
    id: str = Field(alias="_id")

    def tokens(self) -> list[str]:
        return tokenize(self.text)