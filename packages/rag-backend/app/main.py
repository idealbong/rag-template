import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import generate, retrieve
from dotenv import load_dotenv

# Load .env into os.environ at startup
load_dotenv()
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "http://localhost:5173")

app = FastAPI(title="rag-backend")

# CORS 설정 (프론트엔드와의 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_HOST],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG API Server"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Always expose /api/generate
app.include_router(generate.router, prefix="/api")

# Optionally expose /api/retrieve (admin/internal)
if os.getenv("EXPOSE_RETRIEVE_ENDPOINT", "false").lower() == "true":
    app.include_router(retrieve.router, prefix="/api")
