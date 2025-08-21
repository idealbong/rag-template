from fastapi import APIRouter, HTTPException
from app.models import RetrievalRequest, RetrievalResponse, DocumentChunk
from app.services.retrieval_service import RetrievalService
import time
import os

from sentence_transformers import CrossEncoder

router = APIRouter()
retrieval_service = RetrievalService()
cross_encoder = CrossEncoder(os.getenv("CROSS_ENCODER_MODEL", "dragonkue/bge-reranker-v2-m3-ko"))

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    사용자의 질문으로 관련 문서 검색
    """
    try:
        start_time = time.time()
        
        # 1) bi-encoder 기반 후보 검색
        candidate_chunks = await retrieval_service.retrieve(
            question=request.query,
            top_k=request.candidate_k
        )
        
        # 2) reranking 입력 생성
        pairs = [(request.query, chunk.text) for chunk in candidate_chunks]

        # 3) cross-encoder로 점수 계산
        scores = cross_encoder.predict(pairs)  # numpy array

        # 4) DocumentChunk에 새 점수 반영 & 정렬
        reranked = []
        for chunk, score in zip(candidate_chunks, scores):
            # 기존 bi-encoder 점수 대신 cross-encoder 점수 사용
            reranked.append(DocumentChunk(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                url=chunk.url,
                source_type=chunk.source_type,
                score=float(round(score, 4))
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return RetrievalResponse(
            chunks=reranked[:request.top_k],  # 상위 K개만 반환
            elapsed_ms=elapsed_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")
