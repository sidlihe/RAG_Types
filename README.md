# 🚀 Advanced RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with multiple chunking strategies, retrieval techniques, and comprehensive evaluation metrics.

## ✨ Features

### 🧩 Multiple Chunking Strategies
- **Fixed**: Traditional recursive character splitting with configurable size/overlap
- **Semantic**: Groups sentences by semantic similarity using embeddings
- **Proposition**: Splits text into atomic facts and propositions
- **Agentic**: Uses LLM to intelligently determine optimal chunk boundaries

### 🔍 Advanced Retrieval Techniques
- **Dense**: Pure semantic search using vector embeddings
- **Sparse**: BM25 keyword-based search
- **Hybrid**: Ensemble combining dense + sparse retrieval
- **Hybrid + Rerank**: Hybrid with cross-encoder reranking (most accurate)
- **Parent-Child**: Retrieve small chunks for precision, return parent for context
- **Multi-Query**: Generate multiple query variations for better coverage
- **HyDE**: Hypothetical Document Embeddings for complex queries

### 📊 Comprehensive Evaluation
- **RAGAS Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Answer Quality**: METEOR, ROUGE-1, ROUGE-2, ROUGE-L
- **Retrieval Metrics**: MRR, NDCG, Precision@K, Recall@K
- **Performance Tracking**: Latency monitoring, metrics dashboard

### 🎯 Query Optimization
- Automatic query type classification (factual, analytical, summarization, etc.)
- Query expansion with synonyms and variations
- Query rewriting for improved clarity
- Intent detection
- Automatic retrieval strategy selection based on query type

### 💻 User Interface
- Multi-tab Gradio interface
- Real-time metrics display
- Strategy comparison tool
- Conversation export
- Source citation tracking

## 📦 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd adv_langchain
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (for METEOR metric)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

5. **Set up environment variables**

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

## 🚀 Quick Start

### Run the Web Interface
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

### Basic Usage

1. **Upload a PDF** in the "Document Processing" tab
2. **Select chunking strategy** (start with "fixed" for fastest processing)
3. **Click "Process Document"**
4. **Go to "Chat" tab** and ask questions
5. **Select retrieval strategy** (try "hybrid_rerank" for best accuracy)

### Advanced Usage

#### Test Different Strategies
Use the "Compare Strategies" tab to test multiple retrieval methods on the same question.

#### View Metrics
Enable evaluation in the Chat tab and view aggregated metrics in the "Metrics Dashboard" tab.

#### Programmatic Usage
```python
from src.ingestion import load_and_split_pdf
from src.rag_engine import RAGPipeline

# Initialize
pipeline = RAGPipeline(api_key="your-api-key")

# Load and chunk document
chunks = load_and_split_pdf(
    "path/to/document.pdf",
    chunking_strategy="semantic"  # or "fixed", "proposition", "agentic"
)

# Ingest
pipeline.ingest_documents(chunks)

# Query with specific strategy
result = pipeline.query(
    question="What is the main topic?",
    strategy="hybrid_rerank",
    enable_evaluation=True
)

print(result["answer"])
print(result["evaluation"])
```

## 🧪 Testing

Run the automated test suite:

```bash
python tests/test_rag_system.py --pdf data/your_document.pdf --output test_report.md
```

This will test:
- All chunking strategies
- All retrieval strategies
- Evaluation metrics
- Generate a detailed report

## 📊 Strategy Selection Guide

### Chunking Strategies

| Strategy | Best For | Speed | Quality |
|----------|----------|-------|---------|
| Fixed | General purpose, fast processing | ⚡⚡⚡ | ⭐⭐⭐ |
| Semantic | Maintaining topic coherence | ⚡⚡ | ⭐⭐⭐⭐ |
| Proposition | Fact-based documents | ⚡⚡ | ⭐⭐⭐⭐ |
| Agentic | Complex documents, best quality | ⚡ | ⭐⭐⭐⭐⭐ |

### Retrieval Strategies

| Strategy | Best For | Speed | Accuracy |
|----------|----------|-------|----------|
| Dense | Semantic similarity | ⚡⚡⚡ | ⭐⭐⭐ |
| Sparse | Keyword matching | ⚡⚡⚡ | ⭐⭐ |
| Hybrid | Balanced approach | ⚡⚡ | ⭐⭐⭐⭐ |
| Hybrid + Rerank | Best accuracy | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| Parent-Child | Need context | ⚡⚡ | ⭐⭐⭐⭐ |
| Multi-Query | Ambiguous questions | ⚡ | ⭐⭐⭐⭐ |
| HyDE | Complex analysis | ⚡ | ⭐⭐⭐⭐⭐ |

### Query Type → Strategy Mapping

The system can automatically select the best strategy:

- **Factual questions** (who, what, when) → Hybrid + Rerank
- **Analytical questions** (why, how, explain) → HyDE
- **Summarization** → Parent-Child
- **Comparison** → Multi-Query
- **Procedural** (steps, how-to) → Parent-Child

## 📁 Project Structure

```
adv_langchain/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── utils.py                  # Logging utilities
│   ├── ingestion.py              # Document loading & chunking
│   ├── chunking_strategies.py    # Multiple chunking implementations
│   ├── retrieval_strategies.py   # Multiple retrieval implementations
│   ├── query_optimizer.py        # Query classification & optimization
│   ├── evaluation.py             # RAGAS, METEOR, and other metrics
│   └── rag_engine.py             # Main RAG pipeline
├── tests/
│   └── test_rag_system.py        # Automated testing
├── data/                         # PDF storage
├── logs/                         # Application logs
├── app.py                        # Gradio UI
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEMANTIC_SIMILARITY_THRESHOLD = 0.7

# Retrieval
RETRIEVAL_K = 10
RERANK_TOP_N = 3

# Evaluation
ENABLE_RAGAS = True
ENABLE_METEOR = True

# Query Optimization
AUTO_STRATEGY_SELECTION = True
ENABLE_QUERY_CLASSIFICATION = True
```

## 📈 Evaluation Metrics Explained

### RAGAS Metrics
- **Faithfulness**: Is the answer grounded in the retrieved context? (0-1)
- **Answer Relevancy**: Does the answer address the question? (0-1)
- **Context Precision**: Are relevant chunks ranked higher? (0-1)
- **Context Recall**: Was all relevant info retrieved? (0-1)

### Answer Quality
- **METEOR**: Semantic similarity with ground truth (0-1)
- **ROUGE**: N-gram overlap metrics (0-1)

### Retrieval Quality
- **MRR**: Mean Reciprocal Rank - position of first relevant doc
- **NDCG**: Normalized Discounted Cumulative Gain - ranking quality
- **Precision@K**: % of top-K that are relevant
- **Recall@K**: % of relevant docs in top-K

## 🔧 Troubleshooting

### RAGAS not working
```bash
pip install ragas datasets
```

### NLTK data missing
```bash
python -c "import nltk; nltk.download('all')"
```

### ChromaDB errors
Delete the vector database and re-ingest:
```bash
rm -rf data/chroma_db
```

### Out of memory
Reduce chunk size or use fixed chunking:
```python
Config.CHUNK_SIZE = 500
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional chunking strategies
- More retrieval techniques
- Additional evaluation metrics
- Performance optimizations
- UI enhancements

## 📝 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - RAG framework
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Gradio](https://gradio.app/) - UI framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy RAG-ing! 🚀**
