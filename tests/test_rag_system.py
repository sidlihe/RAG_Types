"""
Testing and Benchmarking Module

Automated tests for:
- Chunking strategies
- Retrieval strategies
- Evaluation metrics
- End-to-end RAG pipeline
"""

import os
import sys
from typing import List, Dict, Any
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import load_and_split_pdf
from src.rag_engine import RAGPipeline
from src.evaluation import RAGEvaluator
from src.config import Config
from src.utils import get_logger

logger = get_logger("Testing")


class RAGTester:
    """Test suite for RAG system"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.test_results = []
    
    def test_chunking_strategies(self, pdf_path: str) -> Dict[str, Any]:
        """
        Test all chunking strategies on a PDF.
        
        Args:
            pdf_path: Path to test PDF
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing chunking strategies...")
        
        strategies = ["fixed", "semantic", "proposition"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} chunking...")
            
            try:
                start_time = time.time()
                chunks = load_and_split_pdf(
                    pdf_path,
                    chunking_strategy=strategy,
                    api_key=self.api_key
                )
                duration = time.time() - start_time
                
                # Calculate statistics
                chunk_sizes = [len(c.page_content) for c in chunks]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                
                results[strategy] = {
                    "success": True,
                    "num_chunks": len(chunks),
                    "avg_chunk_size": avg_size,
                    "min_chunk_size": min(chunk_sizes),
                    "max_chunk_size": max(chunk_sizes),
                    "duration_seconds": duration
                }
                
                logger.info(f"{strategy}: {len(chunks)} chunks, avg size {avg_size:.0f}")
                
            except Exception as e:
                logger.error(f"{strategy} failed: {e}")
                results[strategy] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def test_retrieval_strategies(self, 
                                  pdf_path: str,
                                  test_questions: List[str]) -> Dict[str, Any]:
        """
        Test all retrieval strategies.
        
        Args:
            pdf_path: Path to test PDF
            test_questions: List of test questions
        
        Returns:
            Test results dictionary
        """
        logger.info("Testing retrieval strategies...")
        
        # Initialize pipeline
        pipeline = RAGPipeline(api_key=self.api_key)
        
        # Ingest document
        chunks = load_and_split_pdf(pdf_path, chunking_strategy="fixed", api_key=self.api_key)
        pipeline.ingest_documents(chunks)
        
        strategies = ["dense", "sparse", "hybrid", "hybrid_rerank"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} retrieval...")
            
            strategy_results = []
            
            for question in test_questions:
                try:
                    start_time = time.time()
                    result = pipeline.query(
                        question=question,
                        strategy=strategy,
                        enable_optimization=False,
                        enable_evaluation=False
                    )
                    duration = time.time() - start_time
                    
                    strategy_results.append({
                        "question": question,
                        "answer_length": len(result["answer"]),
                        "num_contexts": result["num_contexts"],
                        "latency_ms": duration * 1000
                    })
                    
                except Exception as e:
                    logger.error(f"{strategy} failed on '{question}': {e}")
                    strategy_results.append({
                        "question": question,
                        "error": str(e)
                    })
            
            # Calculate averages
            successful = [r for r in strategy_results if "error" not in r]
            if successful:
                avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
                avg_contexts = sum(r["num_contexts"] for r in successful) / len(successful)
                
                results[strategy] = {
                    "success_rate": len(successful) / len(test_questions),
                    "avg_latency_ms": avg_latency,
                    "avg_contexts_retrieved": avg_contexts,
                    "details": strategy_results
                }
            else:
                results[strategy] = {
                    "success_rate": 0,
                    "details": strategy_results
                }
        
        return results
    
    def test_evaluation_metrics(self, pdf_path: str) -> Dict[str, Any]:
        """
        Test evaluation metrics.
        
        Args:
            pdf_path: Path to test PDF
        
        Returns:
            Test results
        """
        logger.info("Testing evaluation metrics...")
        
        # Initialize
        pipeline = RAGPipeline(api_key=self.api_key)
        chunks = load_and_split_pdf(pdf_path, chunking_strategy="fixed", api_key=self.api_key)
        pipeline.ingest_documents(chunks)
        
        # Test question with ground truth
        test_case = {
            "question": "What is the main summary?",
            "ground_truth": "This is a test ground truth answer."
        }
        
        try:
            result = pipeline.query(
                question=test_case["question"],
                enable_evaluation=True,
                ground_truth=test_case["ground_truth"]
            )
            
            if "evaluation" in result:
                eval_data = result["evaluation"]
                return {
                    "success": True,
                    "metrics_calculated": list(eval_data.keys()),
                    "sample_metrics": {
                        k: v for k, v in eval_data.items() 
                        if v is not None and k in ["faithfulness", "answer_relevancy", "meteor"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No evaluation data in result"
                }
                
        except Exception as e:
            logger.error(f"Evaluation test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_full_test_suite(self, pdf_path: str) -> str:
        """
        Run complete test suite.
        
        Args:
            pdf_path: Path to test PDF
        
        Returns:
            Test report string
        """
        logger.info("=" * 80)
        logger.info("RUNNING FULL TEST SUITE")
        logger.info("=" * 80)
        
        report_lines = ["# RAG System Test Report\n"]
        
        # Test 1: Chunking
        report_lines.append("## 1. Chunking Strategies Test")
        chunking_results = self.test_chunking_strategies(pdf_path)
        
        for strategy, result in chunking_results.items():
            if result["success"]:
                report_lines.append(f"### {strategy.upper()}")
                report_lines.append(f"- ✅ Success")
                report_lines.append(f"- Chunks: {result['num_chunks']}")
                report_lines.append(f"- Avg Size: {result['avg_chunk_size']:.0f} chars")
                report_lines.append(f"- Duration: {result['duration_seconds']:.2f}s\n")
            else:
                report_lines.append(f"### {strategy.upper()}")
                report_lines.append(f"- ❌ Failed: {result['error']}\n")
        
        # Test 2: Retrieval
        report_lines.append("## 2. Retrieval Strategies Test")
        test_questions = [
            "What is the main summary?",
            "List key skills mentioned."
        ]
        
        retrieval_results = self.test_retrieval_strategies(pdf_path, test_questions)
        
        for strategy, result in retrieval_results.items():
            report_lines.append(f"### {strategy.upper()}")
            report_lines.append(f"- Success Rate: {result['success_rate']*100:.0f}%")
            if "avg_latency_ms" in result:
                report_lines.append(f"- Avg Latency: {result['avg_latency_ms']:.0f}ms")
                report_lines.append(f"- Avg Contexts: {result['avg_contexts_retrieved']:.1f}\n")
        
        # Test 3: Evaluation
        report_lines.append("## 3. Evaluation Metrics Test")
        eval_results = self.test_evaluation_metrics(pdf_path)
        
        if eval_results["success"]:
            report_lines.append("- ✅ Evaluation system working")
            report_lines.append(f"- Metrics available: {', '.join(eval_results['metrics_calculated'])}")
            
            if eval_results.get("sample_metrics"):
                report_lines.append("\n**Sample Metrics:**")
                for metric, value in eval_results["sample_metrics"].items():
                    report_lines.append(f"- {metric}: {value:.3f}")
        else:
            report_lines.append(f"- ❌ Failed: {eval_results['error']}")
        
        report = "\n".join(report_lines)
        
        logger.info("\n" + report)
        logger.info("=" * 80)
        logger.info("TEST SUITE COMPLETE")
        logger.info("=" * 80)
        
        return report


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG System")
    parser.add_argument("--pdf", required=True, help="Path to test PDF")
    parser.add_argument("--api-key", help="Google API key (or use .env)")
    parser.add_argument("--output", default="test_report.md", help="Output file for report")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key provided. Use --api-key or set GOOGLE_API_KEY in .env")
        sys.exit(1)
    
    # Run tests
    tester = RAGTester(api_key=api_key)
    report = tester.run_full_test_suite(args.pdf)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Test report saved to: {args.output}")


if __name__ == "__main__":
    main()
