import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.utils import get_logger

logger = get_logger("RAGEngine")

class RAGPipeline:
    def __init__(self, api_key=None):
        self.api_key = api_key if api_key else Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("API Key is missing. Provide it via .env or UI.")
        
        # Initialize Embedding Model (Runs locally)
        logger.info("Loading Embeddings Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # Initialize Reranker Model (Runs locally)
        logger.info("Loading Reranker Model...")
        self.reranker_model = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL)

    def ingest_documents(self, chunks):
        """Stores chunks in ChromaDB."""
        # Optional: Reset DB for new file upload to keep demo clean
        if os.path.exists(Config.CHROMA_DIR):
            shutil.rmtree(Config.CHROMA_DIR)
            
        logger.info("Creating Vector Store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_DIR,
            collection_name="user_pdf_collection"
        )
        logger.info("Vector Store created successfully.")
        return self.vectorstore

    def build_chain(self, chunks):
        """Builds the Hybrid Search + Reranking + LLM Chain."""
        
        # 1. Base Retrievers
        # A. Sparse (Keyword)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = Config.RETRIEVAL_K
        
        # B. Dense (Semantic)
        chroma_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        # 2. Ensemble Retriever (Hybrid)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.4, 0.6] # Slightly favor semantic
        )
        
        # 3. Reranking Logic
        compressor = CrossEncoderReranker(
            model=self.reranker_model, 
            top_n=Config.RERANK_TOP_N
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        # 4. LLM Setup
        llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=0.2,
            google_api_key=self.api_key,
            max_retries=2
        )
        
        # 5. Prompt & Chain
        prompt = ChatPromptTemplate.from_template("""
        You are an intelligent document assistant. Use the context below to answer the user's question.
        If the answer isn't in the context, say "I cannot find the answer in the provided document."

        <context>
        {context}
        </context>

        Question: {input}
        """)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(compression_retriever, document_chain)
        
        return rag_chain

    def query(self, chain, question):
        if not chain:
            return "Please upload a document first."
        try:
            response = chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            logger.error(f"Query Error: {e}")
            return f"Error occurred: {str(e)}"