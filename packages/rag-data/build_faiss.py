import os
import json
import torch
import argparse
from tqdm import tqdm
from slugify import slugify
import multiprocessing as mp

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 설정
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 4  # GPU 메모리에 따라 조정 가능
MAX_WORKERS = min(4, mp.cpu_count())  # CPU 코어 수에 따라 조정
USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
FAISS_DISTANCE_STRATEGY = 'cosine'  # or 'euclidean'

def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_documents(chunks: list[dict]) -> list[Document]:
    documents = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "").strip()
        if len(chunk_text) > 20:
            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "id": chunk.get("id", str(i)),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "source_type": chunk.get("source_type", "Unknown"),
                }
            ))
    return documents

def create_optimized_embeddings(documents, embedding_model, batch_size=BATCH_SIZE):
    """배치 처리로 GPU 사용률을 최적화한 임베딩 생성"""
    print(f"🚀 Creating embeddings with batch size: {batch_size}")
    
    # 텍스트만 추출
    texts = [doc.page_content for doc in documents]
    
    # 배치 단위로 처리
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_embeddings

def main():
    p = argparse.ArgumentParser(description="Build FAISS index from text chunks")
    p.add_argument("--input", type=str, required=True,
                   help="Input JSONL file path containing text chunks")
    p.add_argument("--index_dir", type=str, default=INDEX_DIR,
                   help="Directory to save the FAISS index")
    p.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL_NAME,
                   help="HuggingFace model name for embeddings")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Batch size for embedding generation")
    p.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                   help="Number of parallel workers for processing")
    p.add_argument("--device", type=str, default='cpu',
                   help="Use cuda for GPU acceleration/mps for Mac GPU/cpu for CPU")
    p.add_argument("--faiss_distance_strategy", type=str, default=FAISS_DISTANCE_STRATEGY,
                   help="Distance strategy for FAISS (cosine or euclidean)")
    args = p.parse_args()
    
    print(f"📦 Loading chunks from: {args.input}")
    chunks = load_chunks(args.input)
    documents = build_documents(chunks)
    print(f"✅ Loaded {len(documents)} documents")

    print(f"🤖 Loading embedding model: {args.embedding_model}")
    # GPU 사용 설정 및 최적화
    device = args.device
    print(f"🔧 Using device: {device}")
    
    # GPU 메모리 최적화 설정
    model_kwargs = {
        'device': device,
        'model_kwargs': {
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        }
    }
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model, 
        model_kwargs=model_kwargs
    )

    print("🔍 Creating optimized FAISS index with batch processing...")
    
    # 최적화된 방식으로 임베딩 생성
    embeddings = create_optimized_embeddings(documents, embedding_model, args.batch_size)
    
    # FAISS 인덱스 생성
    print("🏗️ Building FAISS index...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip([doc.page_content for doc in documents], embeddings)),
        embedding=embedding_model,
        metadatas=[doc.metadata for doc in documents],
        distance_strategy=args.faiss_distance_strategy
    )

    os.makedirs(args.index_dir, exist_ok=True)
    vector_store.save_local(args.index_dir)
    print(f"💾 Index saved to: {args.index_dir}/index.faiss and index.pkl")
    print(f"🎯 GPU optimization: Batch size {args.batch_size}, Workers {args.max_workers}")

if __name__ == "__main__":
    main()
