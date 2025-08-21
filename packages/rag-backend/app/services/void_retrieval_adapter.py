from typing import List
from app.models import DocumentChunk

class VoidRetrievalAdapter:
    def __init__(self):
        self._loaded = False

    def load(self) -> None:
        pass

    def retrieve(self, query: str, k: int) -> List[DocumentChunk]:
        return []

    def count(self) -> int:
        return 0