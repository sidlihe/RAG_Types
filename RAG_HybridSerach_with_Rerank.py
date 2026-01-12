import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import custom logger
from logger import get_logger

import traceback
import shutil

# 1. Setup Environment and Logger
load_dotenv()
logger = get_logger("HybridRAG")
model_name = "gemini-2.0-flash"
# Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not found in .env file")
    raise ValueError("Please set GOOGLE_API_KEY in .env")

class HybridSearchPipeline:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        # Validation: Check if file exists immediately
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"The PDF file at {self.pdf_path} was not found.")
            
        self.documents = []
        self.chunks = []
        self.ensemble_retriever = None
        self.llm = None
        self.rag_chain = None
        self.persist_dir = "./chroma_db_data"
        
        # Initialize Embeddings
        logger.info("Initializing HuggingFace Embeddings...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error("Failed to download/load embeddings.")
            raise e

    def validate_llm_connection(self, max_retries=3):
        """
        Robustness Check: Tests the API key and Model availability 
        before doing any heavy lifting. Uses exponential backoff for retries.
        """
        logger.info("Testing connection to Google Gemini 2.0 Flash...")
        
        for attempt in range(max_retries):
            try:
                test_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.1,
                    max_retries=2
                )
                test_llm.invoke("Hello")
                logger.info(" Google Gemini 2.0 Flash connection successful.")
                return True
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f" Failed to connect to Google Gemini after {max_retries} attempts: {e}")
                    logger.error("Suggestion: Check your GOOGLE_API_KEY and ensure you have quota available")
                    raise e

    def load_and_process_documents(self):
        """Loads PDF, cleans metadata, and splits text."""
        logger.info(f"Loading PDF from: {self.pdf_path}")
        
        try:
            loader = PyPDFLoader(self.pdf_path)
            raw_docs = loader.load()
            
            if not raw_docs:
                raise ValueError("PDF loaded but contained no text/pages.")

            # CLEAN METADATA (Crucial for ChromaDB)
            logger.info("Cleaning document metadata...")
            for doc in raw_docs:
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        if isinstance(value, (list, dict)):
                            doc.metadata[key] = str(value)
                        if value is None:
                            doc.metadata[key] = ""

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.chunks = text_splitter.split_documents(raw_docs)
            
            if not self.chunks:
                raise ValueError("Text splitting resulted in 0 chunks.")
                
            logger.info(f"Document split into {len(self.chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise e

    def build_hybrid_retriever(self, alpha: float = 0.5, k: int = 4):
        """
        Builds an Ensemble Retriever (Hybrid Search).
        """
        if not self.chunks:
            raise ValueError("No chunks found. Run load_and_process_documents() first.")

        logger.info(f"Building Hybrid Retriever with Alpha={alpha}...")

        try:
            # 1. Sparse Retriever (BM25 - Keyword Search)
            bm25_retriever = BM25Retriever.from_documents(self.chunks)
            bm25_retriever.k = k

            # 2. Dense Retriever (Chroma - Semantic Search)
            # Cleanup previous DB if exists to ensure fresh start (Robustness)
            if os.path.exists(self.persist_dir):
                try:
                    shutil.rmtree(self.persist_dir)
                except:
                    pass # Ignore permission errors on cleanup

            vectorstore = Chroma.from_documents(
                documents=self.chunks, 
                embedding=self.embeddings,
                collection_name="siddhesh_gemini_collection",
                persist_directory=self.persist_dir # Add persistence
            )
            chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

            # 3. Ensemble (Hybrid)
            sparse_weight = 1.0 - alpha
            dense_weight = alpha
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=[sparse_weight, dense_weight]
            )
            logger.info("Hybrid Retriever built successfully.")
        except Exception as e:
            logger.error(f"Error building retriever: {e}")
            raise e

    def setup_llm_chain(self):
        """Sets up the Gemini 2.0 Flash LLM and the RAG chain."""
        logger.info("Setting up Gemini 2.0 Flash LLM Chain...")
        
        try:
            # Initialize Gemini 2.0 Flash (Experimental)
            # Using gemini-2.0-flash-exp for latest features and performance
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,  # Lower temperature for more factual responses
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                max_retries=3,
                request_timeout=90  # Increased timeout for robustness
            )

            # Create Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant. Answer the question based strictly on the provided context.
            If the answer is not in the context, say "I don't know based on this document."

            <context>
            {context}
            </context>

            Question: {input}
            """)

            # Create Chain
            if self.ensemble_retriever is None:
                raise ValueError("Retriever not initialized.")

            document_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.ensemble_retriever, document_chain)
            logger.info("LLM Chain created successfully.")
            
        except Exception as e:
            logger.error(f"Error setting up LLM chain: {e}")
            raise e

    def query(self, user_question: str, max_retries=3):
        """Invokes the chain with retry logic for robustness."""
        if not self.rag_chain:
            raise ValueError("Chain not initialized.")
        
        logger.info(f"Processing query: {user_question}")
        
        for attempt in range(max_retries):
            try:
                response = self.rag_chain.invoke({"input": user_question})
                logger.info("Response generated successfully.")
                return response["answer"]
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                if attempt < max_retries - 1:
                    logger.warning(f"Query attempt {attempt + 1} failed: {str(e)[:100]}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error during query execution after {max_retries} attempts: {e}")
                    raise e

# --- Execution Block ---
if __name__ == "__main__":
    # Path to your PDF
    # Ensure this path is correct
    pdf_path = r"Siddhesh_Lihe.pdf"
    
    # Initialize Pipeline
    try:
        pipeline = HybridSearchPipeline(pdf_path)
        
        # 0. Test Connection First (Fail Fast)
        pipeline.validate_llm_connection()

        # 1. Load Data
        pipeline.load_and_process_documents()
        
        # 2. Build Hybrid Search
        pipeline.build_hybrid_retriever(alpha=0.5, k=3)
        
        # 3. Setup LLM (Gemini)
        pipeline.setup_llm_chain()
        
        # 4. Ask a Question
        question = "What are the key skills mentioned in the document?"
        print(f"\nQuestion: {question}")
        
        answer = pipeline.query(question)
        print(f"Answer: {answer}")
        
    except Exception as e:
        print("\n--- AN ERROR OCCURRED ---")
        # Print only the message first for clarity, then full trace
        print(f"Error Message: {str(e)}")
        print("\nFull Traceback:")
        traceback.print_exc()