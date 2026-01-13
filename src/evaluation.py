#src/evaluation.py
"""
Comprehensive Evaluation Module for RAG Systems

Implements multiple evaluation metrics:
1. RAGAS Metrics - Faithfulness, Answer Relevancy, Context Precision, Context Recall
2. Answer Quality - METEOR, ROUGE, BERTScore
3. Retrieval Metrics - MRR, NDCG, Precision@K, Recall@K
4. Performance Metrics - Latency, Throughput
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import numpy as np
from langchain_core.documents import Document

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# METEOR and other NLP metrics
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize
    import nltk
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from src.utils import get_logger

logger = get_logger("Evaluation")


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    # RAGAS metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    
    # Answer quality metrics
    meteor: Optional[float] = None
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    
    # Retrieval metrics
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    
    # Performance metrics
    latency_ms: Optional[float] = None
    num_retrieved_docs: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = ["=== Evaluation Results ==="]
        
        if self.faithfulness is not None:
            lines.append(f"Faithfulness: {self.faithfulness:.3f}")
        if self.answer_relevancy is not None:
            lines.append(f"Answer Relevancy: {self.answer_relevancy:.3f}")
        if self.context_precision is not None:
            lines.append(f"Context Precision: {self.context_precision:.3f}")
        if self.meteor is not None:
            lines.append(f"METEOR: {self.meteor:.3f}")
        if self.latency_ms is not None:
            lines.append(f"Latency: {self.latency_ms:.0f}ms")
        
        return "\n".join(lines)


class RAGEvaluator:
    """Main evaluation class for RAG systems"""
    
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        if METEOR_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
    
    def evaluate_ragas(self,
                       question: str,
                       answer: str,
                       contexts: List[str],
                       ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional reference answer for context recall
        
        Returns:
            Dictionary of RAGAS metric scores
        """
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available. Install with: pip install ragas")
            return {}
        
        try:
            # Select metrics based on available data
            # faithfulness and answer_relevancy are usually fine without ground_truth
            # context_precision and context_recall usually REQUIRE ground_truth/reference in newer RAGAS versions
            
            selected_metrics = [faithfulness, answer_relevancy]
            
            # Prepare dataset
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                # In some versions of ragas, 'reference' is the expected column name
                data["reference"] = [ground_truth]
                selected_metrics.append(context_precision)
                selected_metrics.append(context_recall)
                logger.info("Ground truth provided → including context_precision & context_recall")
            else:
                # If no ground truth, we can only do metrics that don't need it
                logger.warning("No ground truth provided. Skipping context_precision and context_recall.")
            
            dataset = Dataset.from_dict(data)
            
            # Evaluate with custom models to avoid OpenAI dependency
            logger.info(f"Running RAGAS evaluation with metrics: {[m.name for m in selected_metrics]}...")
            
            # Wrap models for Ragas
            ragas_llm = LangchainLLMWrapper(self.llm) if self.llm else None
            ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings) if self.embeddings else None
            
            result = evaluate(
                dataset, 
                metrics=selected_metrics, 
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
            
            # Extract results safely - in RAGAS 0.4.2, result is an EvaluationResult object
            # It usually has a .scores attribute which is a list of results per row
            try:
                if hasattr(result, "scores") and len(result.scores) > 0:
                    scores_dict = result.scores[0]
                elif hasattr(result, "to_dict"):
                    scores_dict = result.to_dict()
                else:
                    # Fallback to items() if it behaves like a dict
                    scores_dict = dict(result.items())
            except Exception as e:
                logger.warning(f"Standard indexing failed, trying direct access: {e}")
                scores_dict = result
            
            return {
                "faithfulness": scores_dict.get("faithfulness", 0.0) if isinstance(scores_dict, dict) else getattr(scores_dict, "faithfulness", 0.0),
                "answer_relevancy": scores_dict.get("answer_relevancy", 0.0) if isinstance(scores_dict, dict) else getattr(scores_dict, "answer_relevancy", 0.0),
                "context_precision": (scores_dict.get("context_precision") if isinstance(scores_dict, dict) else getattr(scores_dict, "context_precision", None)) if ground_truth else None,
                "context_recall": (scores_dict.get("context_recall") if isinstance(scores_dict, dict) else getattr(scores_dict, "context_recall", None)) if ground_truth else None
            }
        
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
    
    def evaluate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
        
        Returns:
            METEOR score (0-1)
        """
        if not METEOR_AVAILABLE:
            logger.warning("METEOR not available. Install with: pip install nltk")
            return 0.0
        
        try:
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            
            score = meteor_score([ref_tokens], hyp_tokens)
            return score
        
        except Exception as e:
            logger.error(f"METEOR calculation failed: {e}")
            return 0.0
    
    def evaluate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Ground truth answer
            hypothesis: Generated answer
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE not available. Install with: pip install rouge-score")
            return {}
        
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            
            return {
                "rouge_1": scores['rouge1'].fmeasure,
                "rouge_2": scores['rouge2'].fmeasure,
                "rouge_l": scores['rougeL'].fmeasure
            }
        
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {}
    
    def evaluate_retrieval_mrr(self, 
                               retrieved_docs: List[Document],
                               relevant_doc_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of IDs of relevant documents
        
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", ""))
            if doc_id in relevant_doc_ids:
                return 1.0 / i
        return 0.0
    
    def evaluate_retrieval_ndcg(self,
                                retrieved_docs: List[Document],
                                relevance_scores: List[float],
                                k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevance_scores: Relevance scores for each document (0-1)
            k: Number of top documents to consider
        
        Returns:
            NDCG@K score
        """
        if not relevance_scores:
            return 0.0
        
        # DCG
        dcg = sum(
            (2**rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )
        
        # IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(
            (2**rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval_precision_recall(self,
                                           retrieved_docs: List[Document],
                                           relevant_doc_ids: List[str],
                                           k: int = 10) -> Tuple[float, float]:
        """
        Calculate Precision@K and Recall@K.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of IDs of relevant documents
            k: Number of top documents to consider
        
        Returns:
            Tuple of (precision@k, recall@k)
        """
        top_k = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_retrieved = 0
        for doc in top_k:
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", ""))
            if doc_id in relevant_doc_ids:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        
        return precision, recall
    
    def evaluate_full(self,
                     question: str,
                     answer: str,
                     contexts: List[str],
                     retrieved_docs: List[Document],
                     ground_truth: Optional[str] = None,
                     relevant_doc_ids: Optional[List[str]] = None,
                     latency_ms: Optional[float] = None) -> EvaluationResult:
        """
        Run full evaluation with all available metrics.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context strings
            retrieved_docs: Retrieved document objects
            ground_truth: Optional reference answer
            relevant_doc_ids: Optional list of relevant document IDs
            latency_ms: Optional query latency in milliseconds
        
        Returns:
            EvaluationResult object with all metrics
        """
        result = EvaluationResult()
        
        # RAGAS metrics
        if RAGAS_AVAILABLE and self.llm:
            ragas_scores = self.evaluate_ragas(question, answer, contexts, ground_truth)
            result.faithfulness = ragas_scores.get("faithfulness")
            result.answer_relevancy = ragas_scores.get("answer_relevancy")
            result.context_precision = ragas_scores.get("context_precision")
            result.context_recall = ragas_scores.get("context_recall")
        
        # Answer quality metrics (if ground truth available)
        if ground_truth:
            result.meteor = self.evaluate_meteor(ground_truth, answer)
            
            rouge_scores = self.evaluate_rouge(ground_truth, answer)
            result.rouge_1 = rouge_scores.get("rouge_1")
            result.rouge_2 = rouge_scores.get("rouge_2")
            result.rouge_l = rouge_scores.get("rouge_l")
        
        # Retrieval metrics (if relevant docs provided)
        if relevant_doc_ids:
            result.mrr = self.evaluate_retrieval_mrr(retrieved_docs, relevant_doc_ids)
            precision, recall = self.evaluate_retrieval_precision_recall(
                retrieved_docs, relevant_doc_ids
            )
            result.precision_at_k = precision
            result.recall_at_k = recall
        
        # Performance metrics
        result.latency_ms = latency_ms
        result.num_retrieved_docs = len(retrieved_docs)
        
        logger.info(f"Evaluation complete: {result.get_summary()}")
        return result


class EvaluationBenchmark:
    """Benchmark multiple RAG configurations"""
    
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.results = []
    
    def run_benchmark(self,
                     test_cases: List[Dict[str, Any]],
                     rag_pipeline,
                     strategies_to_test: List[str]) -> Dict[str, List[EvaluationResult]]:
        """
        Run benchmark across multiple strategies.
        
        Args:
            test_cases: List of test cases with questions and ground truths
            rag_pipeline: RAG pipeline instance
            strategies_to_test: List of strategy names to benchmark
        
        Returns:
            Dictionary mapping strategy names to evaluation results
        """
        results = {}
        
        for strategy in strategies_to_test:
            logger.info(f"Benchmarking strategy: {strategy}")
            strategy_results = []
            
            for test_case in test_cases:
                question = test_case["question"]
                ground_truth = test_case.get("ground_truth")
                
                # Run query with timing
                start_time = time.time()
                answer = rag_pipeline.query(question, strategy=strategy)
                latency_ms = (time.time() - start_time) * 1000
                
                # Get contexts and docs
                contexts = test_case.get("contexts", [])
                retrieved_docs = test_case.get("retrieved_docs", [])
                
                # Evaluate
                eval_result = self.evaluator.evaluate_full(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    retrieved_docs=retrieved_docs,
                    ground_truth=ground_truth,
                    latency_ms=latency_ms
                )
                
                strategy_results.append(eval_result)
            
            results[strategy] = strategy_results
        
        return results
    
    def generate_report(self, results: Dict[str, List[EvaluationResult]]) -> str:
        """Generate comparison report"""
        report = ["=" * 80]
        report.append("RAG EVALUATION BENCHMARK REPORT")
        report.append("=" * 80)
        
        for strategy, eval_results in results.items():
            report.append(f"\n### Strategy: {strategy}")
            report.append("-" * 40)
            
            # Calculate averages
            metrics = {}
            for result in eval_results:
                for key, value in result.to_dict().items():
                    if value is not None:
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
            
            for metric, values in metrics.items():
                avg = np.mean(values)
                std = np.std(values)
                report.append(f"{metric}: {avg:.3f} (±{std:.3f})")
        
        return "\n".join(report)
if __name__ == "__main__":
    main()