import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Lightweight, effective reranker
    
    # Paths
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 10  # Get 10 docs from hybrid search
    RERANK_TOP_N = 3  # Rerank and keep top 3 for LLM