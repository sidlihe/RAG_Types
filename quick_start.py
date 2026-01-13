"""
Quick Start Script for Advanced RAG System

This script helps you get started quickly by:
1. Checking dependencies
2. Downloading required NLTK data
3. Verifying configuration
4. Running a simple test
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "langchain",
        "gradio",
        "chromadb",
        "sentence_transformers",
        "nltk",
        "ragas"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!\n")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    print("📥 Downloading NLTK data...")
    
    try:
        import nltk
        
        datasets = ['punkt', 'wordnet', 'omw-1.4']
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
                print(f"  ✅ {dataset} already downloaded")
            except LookupError:
                print(f"  📥 Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"  ✅ {dataset} downloaded")
        
        print("\n✅ NLTK data ready!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_api_key():
    """Check if API key is configured"""
    print("🔑 Checking API key...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        print(f"  ✅ API key found (length: {len(api_key)})\n")
        return True
    else:
        print("  ⚠️  No API key found in .env file")
        print("  Please add GOOGLE_API_KEY=your_key_here to .env\n")
        return False


def run_simple_test():
    """Run a simple test"""
    print("🧪 Running simple test...")
    
    try:
        from src.chunking_strategies import ChunkingStrategyFactory
        from langchain_core.documents import Document
        
        # Create test document
        test_doc = Document(
            page_content="This is a test document. It has multiple sentences. We will test chunking.",
            metadata={"source": "test"}
        )
        
        # Test fixed chunking
        chunker = ChunkingStrategyFactory.create_chunker("fixed", chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_documents([test_doc])
        
        print(f"  ✅ Chunking test passed ({len(chunks)} chunks created)")
        print("\n✅ System is working!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}\n")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 Advanced RAG System - Quick Start")
    print("=" * 60)
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Step 2: Download NLTK data
    download_nltk_data()
    
    # Step 3: Check API key
    has_api_key = check_api_key()
    
    # Step 4: Run test
    run_simple_test()
    
    # Final instructions
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("📚 Next Steps:")
    print()
    print("1. Run the application:")
    print("   python app.py")
    print()
    print("2. Open your browser to:")
    print("   http://127.0.0.1:7860")
    print()
    print("3. Upload a PDF and start asking questions!")
    print()
    
    if not has_api_key:
        print("⚠️  Don't forget to add your Google API key to .env file!")
        print()
    
    print("📖 For more info, see README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
