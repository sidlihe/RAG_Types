"""
Advanced Retrieval Strategies for RAG Systems

Implements multiple retrieval techniques:
1. DenseRetriever - Pure semantic search
2. SparseRetriever - BM25 keyword search  
3. HybridRetriever - Ensemble of dense + sparse
4. ParentChildRetriever - Retrieve small chunks, return parent context
5. MultiQueryRetriever - Generate multiple query variations
6. HyDERetriever - Hypothetical Document Embeddings
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_logger
from src.config import Config

logger = get_logger("RetrievalStrategies")


class BaseRetriever(ABC):
    """Base class for all retrieval strategies"""
    
    @abstractmethod
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        """Create and return a retriever instance"""
        pass


class DenseRetriever(BaseRetriever):
    """Pure semantic search using vector embeddings"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Dense retriever: k={self.k}")
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )


class SparseRetriever(BaseRetriever):
    """BM25 keyword-based search"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Sparse (BM25) retriever: k={self.k}")
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = self.k
        return bm25


class HybridRetriever(BaseRetriever):
    """
    Ensemble of dense and sparse retrievers.
    Combines semantic understanding with keyword matching.
    """
    
    def __init__(self, k: int = 10, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Hybrid retriever: k={self.k}, weights=({self.dense_weight}, {self.sparse_weight})")
        
        # Dense retriever
        dense = vectorstore.as_retriever(search_kwargs={"k": self.k})
        
        # Sparse retriever
        sparse = BM25Retriever.from_documents(chunks)
        sparse.k = self.k
        
        # Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[sparse, dense],
            weights=[self.sparse_weight, self.dense_weight]
        )
        
        return ensemble


class HybridWithReranker(BaseRetriever):
    """
    Hybrid retrieval with cross-encoder reranking.
    This is the most effective approach for accuracy.
    """
    
    def __init__(self, 
                 k: int = 10, 
                 rerank_top_n: int = 3,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.k = k
        self.rerank_top_n = rerank_top_n
        self.reranker_model = reranker_model
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Hybrid + Reranker: k={self.k}, rerank_top_n={self.rerank_top_n}")
        
        # Base hybrid retriever
        hybrid = HybridRetriever(k=self.k).get_retriever(vectorstore, chunks)
        
        # Add reranking
        reranker = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(
            model=reranker,
            top_n=self.rerank_top_n
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid
        )
        
        return compression_retriever


class ParentChildRetriever(BaseRetriever):
    """
    Retrieves small chunks for precision, but returns parent documents for context.
    Best for maintaining context while having precise retrieval.
    """
    
    def __init__(self, 
                 child_chunk_size: int = 400,
                 parent_chunk_size: int = 2000,
                 k: int = 10):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.k = k
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Parent-Child retriever: child={self.child_chunk_size}, parent={self.parent_chunk_size}")
        
        # Create parent splitter
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=200
        )
        
        # Create child splitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=50
        )
        
        # Storage for parent documents
        store = InMemoryStore()
        
        # Create parent-child retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": self.k}
        )
        
        # Add documents
        retriever.add_documents(chunks)
        
        return retriever


class MultiQueryRetriever(BaseRetriever):
    """
    Generates multiple variations of the query to improve retrieval.
    Useful for handling ambiguous or complex questions.
    """
    
    def __init__(self, api_key: str, k: int = 10, num_queries: int = 3):
        self.api_key = api_key
        self.k = k
        self.num_queries = num_queries
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"Multi-Query retriever: k={self.k}, num_queries={self.num_queries}")
        
        # Base retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        
        # LLM for query generation
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=self.api_key
        )
        
        # Multi-query retriever
        multi_query = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        return multi_query


class HyDERetriever(BaseRetriever):
    """
    Hypothetical Document Embeddings (HyDE).
    Generates a hypothetical answer, then searches for similar documents.
    Effective for complex analytical questions.
    """
    
    def __init__(self, api_key: str, k: int = 10):
        self.api_key = api_key
        self.k = k
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_key
        )
    
    def get_retriever(self, vectorstore, chunks: List[Document], **kwargs):
        logger.info(f"HyDE retriever: k={self.k}")
        
        # We'll wrap the base retriever with HyDE logic
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
        
        # Return a custom HyDE retriever wrapper
        return HyDERetrieverWrapper(
            base_retriever=base_retriever,
            llm=self.llm,
            vectorstore=vectorstore,
            k=self.k
        )


class HyDERetrieverWrapper(LangChainBaseRetriever):
    """Wrapper that implements HyDE retrieval logic"""
    
    base_retriever: Any
    llm: Any
    vectorstore: Any
    k: int
    
    def __init__(self, base_retriever, llm, vectorstore, k: int, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "base_retriever", base_retriever)
        object.__setattr__(self, "llm", llm)
        object.__setattr__(self, "vectorstore", vectorstore)
        object.__setattr__(self, "k", k)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Generate hypothetical document and search"""
        
        # Generate hypothetical answer
        hyde_prompt = f"""Given the question: "{query}"
        
Generate a detailed, hypothetical answer as if you had access to the relevant information.
Write in a factual, informative style.

Hypothetical Answer:"""
        
        try:
            response = self.llm.invoke(hyde_prompt)
            hypothetical_doc = response.content
            
            # Search using the hypothetical document
            docs = self.vectorstore.similarity_search(hypothetical_doc, k=self.k)
            
            logger.info(f"HyDE: Generated hypothetical doc, retrieved {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.warning(f"HyDE generation failed, falling back to standard search: {e}")
            return self.base_retriever.get_relevant_documents(query)


class RetrievalStrategyFactory:
    """Factory to create retrieval strategies"""
    
    @staticmethod
    def create_retriever(strategy: str, vectorstore, chunks: List[Document], **kwargs):
        """
        Create a retriever based on strategy name.
        
        Args:
            strategy: One of ['dense', 'sparse', 'hybrid', 'hybrid_rerank', 
                             'parent_child', 'multi_query', 'hyde']
            vectorstore: Vector store instance
            chunks: Document chunks
            **kwargs: Strategy-specific parameters
        """
        strategy = strategy.lower()
        
        if strategy == "dense":
            retriever_obj = DenseRetriever(k=kwargs.get("k", Config.RETRIEVAL_K))
        
        elif strategy == "sparse":
            retriever_obj = SparseRetriever(k=kwargs.get("k", Config.RETRIEVAL_K))
        
        elif strategy == "hybrid":
            retriever_obj = HybridRetriever(
                k=kwargs.get("k", Config.RETRIEVAL_K),
                dense_weight=kwargs.get("dense_weight", 0.6),
                sparse_weight=kwargs.get("sparse_weight", 0.4)
            )
        
        elif strategy == "hybrid_rerank":
            retriever_obj = HybridWithReranker(
                k=kwargs.get("k", Config.RETRIEVAL_K),
                rerank_top_n=kwargs.get("rerank_top_n", Config.RERANK_TOP_N),
                reranker_model=kwargs.get("reranker_model", Config.RERANKER_MODEL)
            )
        
        elif strategy == "parent_child":
            retriever_obj = ParentChildRetriever(
                child_chunk_size=kwargs.get("child_chunk_size", 400),
                parent_chunk_size=kwargs.get("parent_chunk_size", 2000),
                k=kwargs.get("k", Config.RETRIEVAL_K)
            )
        
        elif strategy == "multi_query":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key required for multi-query retrieval")
            retriever_obj = MultiQueryRetriever(
                api_key=api_key,
                k=kwargs.get("k", Config.RETRIEVAL_K),
                num_queries=kwargs.get("num_queries", 3)
            )
        
        elif strategy == "hyde":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key required for HyDE retrieval")
            retriever_obj = HyDERetriever(
                api_key=api_key,
                k=kwargs.get("k", Config.RETRIEVAL_K)
            )
        
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        return retriever_obj.get_retriever(vectorstore, chunks, **kwargs)
