#src/ingestion.py
import os
from typing import List, Optional
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

try:
    from src.chunking_strategies import ChunkingStrategyFactory
    from src.utils import get_logger
    from src.config import Config
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunking_strategies import ChunkingStrategyFactory
    from utils import get_logger
    from config import Config

logger = get_logger("Ingestion")

def load_and_split_pdf(pdf_path: str, 
                       chunking_strategy: str = "fixed",
                       api_key: Optional[str] = None,
                       **kwargs) -> List[Document]:
    """
    Loads a PDF and splits it into chunks using specified strategy.
    
    Args:
        pdf_path: Path to PDF file
        chunking_strategy: One of ['fixed', 'semantic', 'proposition', 'agentic']
        api_key: Required for agentic chunking
        **kwargs: Additional strategy-specific parameters
    
    Returns:
        List of Document chunks
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    logger.info(f"Processing PDF: {pdf_path} with {chunking_strategy} chunking")
    
    try:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        
        if not raw_docs:
            raise ValueError("PDF is empty.")

        # 2. Clean Metadata (Fixes ChromaDB errors)
        for doc in raw_docs:
            if doc.metadata:
                # Add source information
                doc.metadata["source_file"] = os.path.basename(pdf_path)
                doc.metadata["source_path"] = pdf_path
                
                # Clean invalid values
                for key, value in list(doc.metadata.items()):
                    if value is None:
                        doc.metadata[key] = ""
                    elif isinstance(value, (list, dict)):
                        doc.metadata[key] = str(value)

        # 3. Split using selected strategy
        chunker = ChunkingStrategyFactory.create_chunker(
            strategy=chunking_strategy,
            api_key=api_key,
            **kwargs
        )
        
        chunks = chunker.chunk_documents(raw_docs)
        
        # 4. Enrich metadata
        chunks = _enrich_metadata(chunks, pdf_path, chunking_strategy)
        
        logger.info(f"Successfully split PDF into {len(chunks)} chunks using {chunking_strategy} strategy")
        return chunks

    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise e


def _enrich_metadata(chunks: List[Document], 
                    pdf_path: str,
                    strategy: str) -> List[Document]:
    """
    Enrich chunk metadata with additional information.
    
    Args:
        chunks: List of document chunks
        pdf_path: Source PDF path
        strategy: Chunking strategy used
    
    Returns:
        Chunks with enriched metadata
    """
    for i, chunk in enumerate(chunks):
        # Add unique document ID
        chunk.metadata["doc_id"] = f"{os.path.basename(pdf_path)}_{strategy}_{i}"
        
        # Add chunk statistics
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
        chunk.metadata["word_count"] = len(chunk.page_content.split())
        
        # Add processing timestamp
        from datetime import datetime
        chunk.metadata["processed_at"] = datetime.now().isoformat()
    
    return chunks


def load_multiple_pdfs(pdf_paths: List[str],
                      chunking_strategy: str = "fixed",
                      api_key: Optional[str] = None,
                      **kwargs) -> List[Document]:
    """
    Load and chunk multiple PDF files.
    
    Args:
        pdf_paths: List of PDF file paths
        chunking_strategy: Chunking strategy to use
        api_key: API key for agentic chunking
        **kwargs: Additional parameters
    
    Returns:
        Combined list of chunks from all PDFs
    """
    all_chunks = []
    
    for pdf_path in pdf_paths:
        try:
            chunks = load_and_split_pdf(pdf_path, chunking_strategy, api_key, **kwargs)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue
    
    logger.info(f"Processed {len(pdf_paths)} PDFs, total {len(all_chunks)} chunks")
    return all_chunks

# ────────────────────────────────────────────────
#                  Standalone testing
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    def show_chunk_preview(chunks: List[Document], max_chars: int = 280):
        print(f"\n{'═' * 70}")
        print(f"  {len(chunks)} chunks created")
        print(f"{'═' * 70}\n")

        for i, chunk in enumerate(chunks[:12], 1):  # show first 12 chunks
            text = chunk.page_content.strip().replace("\n", " ")
            preview = text[:max_chars]
            if len(text) > max_chars:
                preview += "..."

            meta = chunk.metadata
            print(f"Chunk {i:2d}   |  {meta.get('char_count',0):5,} chars  |  {preview}")
            print(f"           |  id: {meta.get('doc_id','—')}")
            print(f"           |  words: {meta.get('word_count',0):3}   index: {meta['chunk_index']}/{meta['total_chunks']}")
            print("─" * 90)

    parser = argparse.ArgumentParser(description="Test PDF ingestion & chunking")
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument("--strategy", "-s",
                        default="fixed",
                        choices=["fixed", "semantic", "proposition", "agentic"],
                        help="chunking strategy")
    parser.add_argument("--size", type=int, default=1000,
                        help="chunk size (used by fixed & some others)")
    parser.add_argument("--overlap", type=int, default=180,
                        help="chunk overlap (fixed strategy)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (needed for agentic)")
    parser.add_argument("--max-preview", type=int, default=280,
                        help="max characters to show per chunk preview")

    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()

    if not pdf_path.is_file():
        print(f"Error: Not a file → {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing file:  {pdf_path}")
    print(f"Strategy:         {args.strategy}")
    print(f"Size / overlap:   {args.size} / {args.overlap}")
    print(f"Started:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        if args.strategy == "fixed":
            chunks = load_and_split_pdf(
                pdf_path,
                chunking_strategy="fixed",
                chunk_size=args.size,
                chunk_overlap=args.overlap,
            )
        elif args.strategy == "agentic":
            if not args.api_key:
                print("Error: --api-key required for agentic strategy", file=sys.stderr)
                sys.exit(1)
            chunks = load_and_split_pdf(
                pdf_path,
                chunking_strategy="agentic",
                api_key=args.api_key,
                max_chunk_size=args.size,
            )
        else:
            # semantic, proposition, etc.
            chunks = load_and_split_pdf(
                pdf_path,
                chunking_strategy=args.strategy,
                max_chunk_size=args.size,
            )

        show_chunk_preview(chunks, args.max_preview)

        print(f"\nDone. Total chunks: {len(chunks)}")

    except Exception as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)