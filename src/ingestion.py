import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_logger
from src.config import Config

logger = get_logger("Ingestion")

def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF and splits it into chunks.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # 1. Load
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        
        if not raw_docs:
            raise ValueError("PDF is empty.")

        # 2. Clean Metadata (Fixes ChromaDB errors)
        for doc in raw_docs:
            if doc.metadata:
                for key, value in doc.metadata.items():
                    if value is None:
                        doc.metadata[key] = ""
                    elif isinstance(value, (list, dict)):
                        doc.metadata[key] = str(value)

        # 3. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(raw_docs)
        
        logger.info(f"Successfully split PDF into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise e