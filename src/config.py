import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Paths
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluations")
    
    # Chunking Strategies Configuration
    # Fixed chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Semantic chunking
    SEMANTIC_SIMILARITY_THRESHOLD = 0.7
    SEMANTIC_MAX_CHUNK_SIZE = 1500
    
    # Proposition chunking
    MAX_PROPOSITIONS_PER_CHUNK = 5
    
    # Agentic chunking
    AGENTIC_MAX_CHUNK_SIZE = 1500
    
    # Retrieval Configuration
    RETRIEVAL_K = 10  # Number of documents to retrieve
    RERANK_TOP_N = 3  # Top N after reranking
    
    # Parent-Child retrieval
    PARENT_CHUNK_SIZE = 2000
    CHILD_CHUNK_SIZE = 400
    
    # Multi-Query retrieval
    NUM_QUERY_VARIATIONS = 3
    
    # Hybrid retrieval weights
    DENSE_WEIGHT = 0.6
    SPARSE_WEIGHT = 0.4
    
    # Evaluation Configuration
    ENABLE_RAGAS = True
    ENABLE_METEOR = True
    ENABLE_ROUGE = True
    
    # Performance Configuration
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    MAX_CONCURRENT_QUERIES = 5
    
    # Query Optimization
    ENABLE_QUERY_CLASSIFICATION = True
    ENABLE_QUERY_EXPANSION = False  # Can be slow
    ENABLE_QUERY_REWRITING = False  # Can be slow
    AUTO_STRATEGY_SELECTION = True  # Auto-select retrieval strategy based on query type
    
    # Logging
    LOG_LEVEL = "INFO"
    ENABLE_METRICS_TRACKING = True
    
    # UI Configuration
    GRADIO_THEME = "soft"
    ENABLE_METRICS_DASHBOARD = True
    SHOW_SOURCE_CITATIONS = True