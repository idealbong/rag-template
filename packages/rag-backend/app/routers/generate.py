from fastapi import APIRouter, HTTPException
from app.models import GenerateRequest, GenerateResponse
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
import time

router = APIRouter()
retrieval_service = RetrievalService()
llm_service = LLMService()

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    사용자 질문에 대한 LLM 응답 생성 (RAG 옵션 포함)
    """
    try:
        start_time = time.time()
        
        context_chunks = []
        reference_documents = []
        
        if request.use_rag:
            # RAG를 사용하는 경우 관련 문서 검색
            search_results = await retrieval_service.retrieve(
                question=request.query,
                top_k=5
            )
            context_chunks = [chunk.chunk_text for chunk in search_results]
            reference_documents = search_results  # 참고 문서 정보 저장
        
        # 프롬프트 생성
        prompt = llm_service.create_prompt(
            question=request.query,
            context_chunks=context_chunks
        )
        
        # LLM을 이용한 답변 생성
        response = await llm_service.generate(prompt, max_tokens=request.max_tokens)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return GenerateResponse(
            response=response,
            reference_documents=reference_documents,
            prompt=prompt.to_string(),
            question=request.query,
            elapsed_ms=elapsed_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")
