import os
import asyncio
from typing import List, Protocol, Optional
from app.models import DocumentChunk


# ───────────────────────────────────────────────────────────────
# Vector store adapters (교체 가능)
# ───────────────────────────────────────────────────────────────
class VectorStoreAdapter(Protocol):
    """All adapters must return (Document-like, similarity[0..1]) pairs."""
    def load(self) -> None: ...
    def retrieve(self, query: str, k: int) -> List[DocumentChunk]: ...
    def count(self) -> int: ...


def make_vector_store_adapter() -> VectorStoreAdapter | None:
    vector_db = os.getenv("VECTOR_DB", "none").lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    if vector_db == "faiss":
        from app.services.faiss_adapter import FaissAdapter
        index_dir = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
        return FaissAdapter(index_dir=index_dir, embedding_model_name=embedding_model)
    print(f"Unsupported VECTOR_DB: {vector_db}")
    return None

# ───────────────────────────────────────────────────────────────
# RetrievalService → List[DocumentChunk] 반환
# ───────────────────────────────────────────────────────────────
class RetrievalService:
    def __init__(self, vector_store: Optional[VectorStoreAdapter] = None):
        self.vector_store = vector_store or make_vector_store_adapter()
        if not self.vector_store:
            self._initialize()

    def _initialize(self):
        try:
            print("🔌 Loading vector store adapter...")
            self.vector_store.load()
            print(f"✅ Vector store loaded. count={self.vector_store.count()}")
        except Exception as e:
            print(f"❌ Error initializing SearchService: {e}")
            self.vector_store = None

    async def retrieve(self, question: str, top_k: int = 5) -> List[DocumentChunk]:
        """질문과 유사한 문서 청크를 검색하고, DocumentChunk 리스트로 반환."""
        if not self.vector_store:
            return []
        try:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self.vector_store.retrieve, question, top_k)
            if not chunks:
                print("No similar chunks found.")
                return []
            return chunks
        except Exception as e:
            raise RuntimeError(f"문서 검색 중 오류 발생: {str(e)}")

    def get_index_info(self) -> dict:
        if not self.vector_store:
            return {"status": "not_loaded", "count": 0}
        try:
            return {
                "status": "loaded",
                "count": self.vector_store.count(),
                "backend": os.getenv("VECTOR_DB", "faiss"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
