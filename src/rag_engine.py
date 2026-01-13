"""
Advanced RAG Engine with Multiple Strategies and Evaluation

Features:
- Multiple chunking strategies
- Multiple retrieval strategies  
- Automatic strategy selection
- Evaluation metrics integration
- Caching and performance optimization
- Source citation tracking
"""

import os
import shutil
import time
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.config import Config
from src.utils import get_logger
from src.retrieval_strategies import RetrievalStrategyFactory
from src.query_optimizer import QueryOptimizer, QueryType
from src.evaluation import RAGEvaluator, EvaluationResult

logger = get_logger("RAGEngine")


class RAGPipeline:
    """Advanced RAG Pipeline with multiple strategies and evaluation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key else Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("API Key is missing. Provide it via .env or UI.")
        
        # Initialize components
        logger.info("Initializing Advanced RAG Pipeline...")
        
        # Embedding Model
        logger.info(f"Loading Embeddings Model: {Config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=0.2,
            google_api_key=self.api_key,
            max_retries=2
        )
        
        # Query Optimizer
        self.query_optimizer = QueryOptimizer(api_key=self.api_key)
        
        # Evaluator
        self.evaluator = RAGEvaluator(llm=self.llm)
        
        # State
        self.vectorstore = None
        self.chunks = None
        self.current_retrieval_strategy = "hybrid_rerank"
        self.metrics_history = []
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(self, chunks: List[Document], reset_db: bool = True):
        """
        Store chunks in vector database.
        
        Args:
            chunks: Document chunks to ingest
            reset_db: Whether to reset existing database
        """
        if reset_db and os.path.exists(Config.CHROMA_DIR):
            logger.info("Resetting vector database...")
            shutil.rmtree(Config.CHROMA_DIR)
        
        logger.info(f"Creating Vector Store with {len(chunks)} chunks...")
        self.chunks = chunks
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_DIR,
            collection_name="advanced_rag_collection"
        )
        
        logger.info("Vector Store created successfully")
        return self.vectorstore
    
    def set_retrieval_strategy(self, strategy: str):
        """
        Set the retrieval strategy.
        
        Args:
            strategy: One of ['dense', 'sparse', 'hybrid', 'hybrid_rerank', 
                             'parent_child', 'multi_query', 'hyde']
        """
        self.current_retrieval_strategy = strategy
        logger.info(f"Retrieval strategy set to: {strategy}")
    
    def build_chain(self, 
                   retrieval_strategy: Optional[str] = None,
                   custom_prompt: Optional[str] = None):
        """
        Build RAG chain with specified retrieval strategy.
        
        Args:
            retrieval_strategy: Retrieval strategy to use
            custom_prompt: Optional custom prompt template
        
        Returns:
            RAG chain
        """
        if not self.vectorstore or not self.chunks:
            raise ValueError("Must ingest documents first")
        
        strategy = retrieval_strategy or self.current_retrieval_strategy
        
        logger.info(f"Building RAG chain with {strategy} retrieval...")
        
        # Create retriever
        retriever = RetrievalStrategyFactory.create_retriever(
            strategy=strategy,
            vectorstore=self.vectorstore,
            chunks=self.chunks,
            api_key=self.api_key,
            k=Config.RETRIEVAL_K,
            rerank_top_n=Config.RERANK_TOP_N
        )
        
        # Create prompt
        if custom_prompt:
            prompt = ChatPromptTemplate.from_template(custom_prompt)
        else:
            prompt = self._get_default_prompt()
        
        # Build chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG chain built successfully")
        return rag_chain
    
    def _get_default_prompt(self) -> ChatPromptTemplate:
        """Get default prompt template"""
        template = """You are an intelligent document assistant. Use the context below to answer the user's question accurately and comprehensively.

Guidelines:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say "I cannot find the answer in the provided document."
- Be specific and cite relevant information
- Use clear, professional language

<context>
{context}
</context>

Question: {input}

Answer:"""
        return ChatPromptTemplate.from_template(template)
    
    def query(self, 
             question: str,
             strategy: Optional[str] = None,
             enable_optimization: bool = True,
             enable_evaluation: bool = False,
             ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system with advanced features.
        
        Args:
            question: User question
            strategy: Optional retrieval strategy override
            enable_optimization: Enable query optimization
            enable_evaluation: Enable response evaluation
            ground_truth: Optional ground truth for evaluation
        
        Returns:
            Dictionary with answer, contexts, metrics, etc.
        """
        start_time = time.time()
        
        # Query optimization
        optimized_query = question
        query_type = None
        
        if enable_optimization and Config.ENABLE_QUERY_CLASSIFICATION:
            opt_result = self.query_optimizer.optimize(
                query=question,
                enable_classification=True,
                enable_expansion=Config.ENABLE_QUERY_EXPANSION,
                enable_rewriting=Config.ENABLE_QUERY_REWRITING
            )
            
            optimized_query = opt_result["optimized_query"]
            query_type = opt_result.get("query_type")
            
            # Auto-select strategy based on query type
            if Config.AUTO_STRATEGY_SELECTION and query_type and not strategy:
                strategy = self.query_optimizer.get_retrieval_strategy_recommendation(query_type)
                logger.info(f"Auto-selected strategy: {strategy} for query type: {query_type.value}")
        
        # Build chain with selected strategy
        chain = self.build_chain(retrieval_strategy=strategy)
        
        # Execute query
        try:
            response = chain.invoke({"input": optimized_query})
            answer = response["answer"]
            contexts = [doc.page_content for doc in response.get("context", [])]
            retrieved_docs = response.get("context", [])
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "answer": f"Error occurred: {str(e)}",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Build result
        result = {
            "answer": answer,
            "question": question,
            "optimized_query": optimized_query,
            "query_type": query_type.value if query_type else None,
            "strategy_used": strategy or self.current_retrieval_strategy,
            "contexts": contexts,
            "retrieved_docs": retrieved_docs,
            "num_contexts": len(contexts),
            "latency_ms": latency_ms,
            "sources": self._extract_sources(retrieved_docs)
        }
        
        # Evaluation
        if enable_evaluation and Config.ENABLE_METRICS_TRACKING:
            eval_result = self.evaluator.evaluate_full(
                question=question,
                answer=answer,
                contexts=contexts,
                retrieved_docs=retrieved_docs,
                ground_truth=ground_truth,
                latency_ms=latency_ms
            )
            
            result["evaluation"] = eval_result.to_dict()
            self.metrics_history.append(eval_result)
        
        logger.info(f"Query completed in {latency_ms:.0f}ms")
        return result
    
    def query_simple(self, question: str, strategy: Optional[str] = None) -> str:
        """
        Simple query interface that returns just the answer string.
        
        Args:
            question: User question
            strategy: Optional retrieval strategy
        
        Returns:
            Answer string
        """
        result = self.query(question, strategy=strategy, enable_optimization=True)
        return result["answer"]
    
    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in docs:
            source_info = {
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "chunk_id": doc.metadata.get("doc_id", "Unknown"),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        return sources
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics collected.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        import numpy as np
        
        summary = {}
        
        # Aggregate each metric
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", 
                           "meteor", "latency_ms"]:
            values = [
                getattr(m, metric_name) 
                for m in self.metrics_history 
                if getattr(m, metric_name) is not None
            ]
            
            if values:
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """
        Export metrics history to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        metrics_data = [m.to_dict() for m in self.metrics_history]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def compare_strategies(self, 
                          question: str,
                          strategies: List[str],
                          ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple retrieval strategies on the same question.
        
        Args:
            question: Question to test
            strategies: List of strategies to compare
            ground_truth: Optional ground truth answer
        
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies...")
        
        results = {}
        
        for strategy in strategies:
            result = self.query(
                question=question,
                strategy=strategy,
                enable_evaluation=True,
                ground_truth=ground_truth
            )
            results[strategy] = result
        
        # Generate comparison summary
        comparison = {
            "question": question,
            "strategies_compared": strategies,
            "results": results,
            "summary": self._generate_comparison_summary(results)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable comparison summary"""
        lines = ["=== Strategy Comparison ===\n"]
        
        for strategy, result in results.items():
            lines.append(f"Strategy: {strategy}")
            lines.append(f"  Latency: {result['latency_ms']:.0f}ms")
            
            if "evaluation" in result:
                eval_data = result["evaluation"]
                if eval_data.get("faithfulness"):
                    lines.append(f"  Faithfulness: {eval_data['faithfulness']:.3f}")
                if eval_data.get("answer_relevancy"):
                    lines.append(f"  Relevancy: {eval_data['answer_relevancy']:.3f}")
            
            lines.append("")
        
        return "\n".join(lines)