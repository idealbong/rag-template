import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.models import DocumentChunk

class FaissAdapter:
    def __init__(self, index_dir: str, embedding_model_name: str, use_cuda_env: str = "USE_CUDA"):
        self.vector_db = 'faiss'
        self.embedding_model_name = embedding_model_name
        self._FAISS = FAISS
        self._index_dir = index_dir
        self._emb = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if os.environ.get(use_cuda_env, 'false').lower() == 'true' else 'cpu'}
        )
        self._store: Optional[FAISS] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self._index_dir):
            raise FileNotFoundError(f"FAISS index not found at: {self._index_dir}")
        self._store = self._FAISS.load_local(
            self._index_dir,
            self._emb,
            allow_dangerous_deserialization=True,
            distance_strategy="cosine"  # or "euclidean"
        )
        self._loaded = True
        
    def _ensure_loaded(self):
        if not self._store:
            raise RuntimeError("FAISS store not loaded. Call load() first.")
        
    def _to_score(self, distance: float) -> float:
        """Convert distance to a normalized score [0..1]."""
        # 거리 전략에 따라 변환 (기본: Cosine 가정)
        strategy = getattr(self._store, "distance_strategy", None) if self._store else None
        # COSINE: distance ≈ 1 - cos_sim  (cos_sim ∈ [-1,1])
        # => cos_sim = 1 - distance; norm = (cos_sim + 1)/2 ∈ [0,1]
        if str(strategy).upper().endswith("COSINE"):
            cos_sim = 1.0 - float(distance)
            norm = (cos_sim + 1.0) / 2.0
            return float(max(0.0, min(1.0, norm)))
        # 기타 전략은 보수적으로 랭킹 전용으로만 사용 (0~1 매핑 단순화)
        # 작은 distance가 더 가깝다는 가정 하에
        return 1.0 / (1.0 + float(distance))

    def retrieve(self, query: str, k: int) -> List[DocumentChunk]:
        self._ensure_loaded()
        doc_distance = self._store.similarity_search_with_score(query, k=k)
        chunks: List[DocumentChunk] = []
        for doc, distance in doc_distance:
            score = self._to_score(distance)
            chunk = DocumentChunk(
                chunk_text=doc.page_content,
                chunk_index=doc.metadata.get('chunk_index', 0),
                title=doc.metadata.get('title', 'Unknown'),
                url=doc.metadata.get('url', ''),
                source_type=doc.metadata.get('source_type', 'Unknown'),
                score=round(score, 4)
            )
            chunks.append(chunk)
        return chunks

    def count(self) -> int:
        return int(self._store.index.ntotal if self._store else 0)