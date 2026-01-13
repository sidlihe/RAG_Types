"""
Test all imports to identify any remaining issues
"""

print("Testing imports...")

try:
    print("1. Testing ingestion...")
    from src.ingestion import load_and_split_pdf
    print("   ✅ ingestion OK")
except Exception as e:
    print(f"   ❌ ingestion FAILED: {e}")

try:
    print("2. Testing chunking_strategies...")
    from src.chunking_strategies import ChunkingStrategyFactory
    print("   ✅ chunking_strategies OK")
except Exception as e:
    print(f"   ❌ chunking_strategies FAILED: {e}")

try:
    print("3. Testing retrieval_strategies...")
    from src.retrieval_strategies import RetrievalStrategyFactory
    print("   ✅ retrieval_strategies OK")
except Exception as e:
    print(f"   ❌ retrieval_strategies FAILED: {e}")

try:
    print("4. Testing evaluation...")
    from src.evaluation import RAGEvaluator
    print("   ✅ evaluation OK")
except Exception as e:
    print(f"   ❌ evaluation FAILED: {e}")

try:
    print("5. Testing query_optimizer...")
    from src.query_optimizer import QueryOptimizer
    print("   ✅ query_optimizer OK")
except Exception as e:
    print(f"   ❌ query_optimizer FAILED: {e}")

try:
    print("6. Testing rag_engine...")
    from src.rag_engine import RAGPipeline
    print("   ✅ rag_engine OK")
except Exception as e:
    print(f"   ❌ rag_engine FAILED: {e}")

try:
    print("7. Testing app...")
    import app
    print("   ✅ app OK")
except Exception as e:
    print(f"   ❌ app FAILED: {e}")

print("\n✅ All imports successful! Ready to run app.py")
