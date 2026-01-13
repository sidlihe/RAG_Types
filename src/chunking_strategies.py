#src/chunking_strategies.py
"""
Advanced Chunking Strategies for RAG Systems

This module implements multiple intelligent chunking strategies:
1. FixedChunker - Traditional recursive character splitting
2. SemanticChunker - Groups sentences by semantic similarity
3. PropositionChunker - Splits into atomic propositions/facts
4. AgenticChunker - Uses LLM to intelligently determine boundaries
5. HybridChunker - Combines multiple strategies
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import numpy as np
from src.utils import get_logger
from src.config import Config

logger = get_logger("ChunkingStrategies")

class BaseChunker(ABC):
    """Base class for all chunking strategies"""
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        pass
    
    def _add_metadata(self, chunks: List[Document], strategy_name: str) -> List[Document]:
        """Add chunking metadata to documents"""
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunking_strategy": strategy_name,
                "chunk_size": len(chunk.page_content)
            })
        return chunks


class FixedChunker(BaseChunker):
    """Traditional fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Fixed chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")
        chunks = self.splitter.split_documents(documents)
        return self._add_metadata(chunks, "fixed")


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on sentence embeddings.
    Groups sentences together when semantic similarity is high.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 max_chunk_size: int = 1500):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Semantic chunking: threshold={self.similarity_threshold}")
        all_chunks = []
        
        for doc in documents:
            # Split into sentences
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue
            
            # Get embeddings
            embeddings = self.model.encode(sentences)
            
            # Group by similarity
            chunks = self._group_by_similarity(sentences, embeddings, doc.metadata)
            all_chunks.extend(chunks)
        
        return self._add_metadata(all_chunks, "semantic")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_by_similarity(self, sentences: List[str], 
                            embeddings: np.ndarray,
                            metadata: Dict) -> List[Document]:
        """Group sentences into chunks based on semantic similarity"""
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            # Check if we should continue current chunk
            if (similarity >= self.similarity_threshold and 
                current_size + len(sentences[i]) < self.max_chunk_size):
                current_chunk.append(sentences[i])
                current_size += len(sentences[i])
            else:
                # Create new chunk
                chunks.append(Document(
                    page_content=" ".join(current_chunk),
                    metadata=metadata.copy()
                ))
                current_chunk = [sentences[i]]
                current_size = len(sentences[i])
        
        # Add final chunk
        if current_chunk:
            chunks.append(Document(
                page_content=" ".join(current_chunk),
                metadata=metadata.copy()
            ))
        
        return chunks


class PropositionChunker(BaseChunker):
    """
    Splits text into atomic propositions (single facts/statements).
    Uses simple heuristics to identify proposition boundaries.
    """
    
    def __init__(self, max_propositions_per_chunk: int = 5):
        self.max_propositions_per_chunk = max_propositions_per_chunk
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Proposition chunking: max_props={self.max_propositions_per_chunk}")
        all_chunks = []
        
        for doc in documents:
            propositions = self._extract_propositions(doc.page_content)
            chunks = self._group_propositions(propositions, doc.metadata)
            all_chunks.extend(chunks)
        
        return self._add_metadata(all_chunks, "proposition")
    
    def _extract_propositions(self, text: str) -> List[str]:
        """Extract atomic propositions from text"""
        import re
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        propositions = []
        
        for sentence in sentences:
            # Further split compound sentences
            # Look for conjunctions, semicolons, etc.
            sub_props = re.split(r'[;,]\s+(?:and|but|or|however|moreover|furthermore)\s+', 
                               sentence, flags=re.IGNORECASE)
            propositions.extend([p.strip() for p in sub_props if p.strip()])
        
        return propositions
    
    def _group_propositions(self, propositions: List[str], 
                           metadata: Dict) -> List[Document]:
        """Group propositions into chunks"""
        chunks = []
        
        for i in range(0, len(propositions), self.max_propositions_per_chunk):
            chunk_props = propositions[i:i + self.max_propositions_per_chunk]
            chunks.append(Document(
                page_content=" ".join(chunk_props),
                metadata=metadata.copy()
            ))
        
        return chunks


class AgenticChunker(BaseChunker):
    """
    Uses LLM to intelligently determine chunk boundaries.
    Most sophisticated but slowest approach.
    """
    
    def __init__(self, api_key: str, max_chunk_size: int = 1500):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        self.max_chunk_size = max_chunk_size
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info("Agentic chunking: Using LLM for intelligent boundaries")
        all_chunks = []
        
        for doc in documents:
            chunks = self._llm_chunk(doc)
            all_chunks.extend(chunks)
        
        return self._add_metadata(all_chunks, "agentic")
    
    def _llm_chunk(self, document: Document) -> List[Document]:
        """Use LLM to determine optimal chunk boundaries"""
        text = document.page_content
        
        # If text is small enough, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [document]
        
        # Use LLM to identify logical sections
        prompt = f"""Analyze the following text and identify logical section boundaries.
Return ONLY the section titles or first few words of each section, separated by '|||'.

Text:
{text[:3000]}  # Limit for API

Section boundaries (format: "Section 1|||Section 2|||Section 3"):"""
        
        try:
            response = self.llm.invoke(prompt)
            boundaries = response.content.split("|||")
            
            # Use boundaries to split text
            chunks = self._split_by_boundaries(text, boundaries, document.metadata)
            return chunks
        except Exception as e:
            logger.warning(f"Agentic chunking failed, falling back to fixed: {e}")
            # Fallback to fixed chunking
            fallback = FixedChunker(self.max_chunk_size, 200)
            return fallback.chunk_documents([document])
    
    def _split_by_boundaries(self, text: str, boundaries: List[str], 
                            metadata: Dict) -> List[Document]:
        """Split text using identified boundaries"""
        chunks = []
        current_pos = 0
        
        for boundary in boundaries:
            boundary = boundary.strip()
            if not boundary:
                continue
            
            # Find boundary in text
            pos = text.find(boundary, current_pos)
            if pos > current_pos:
                chunk_text = text[current_pos:pos].strip()
                if chunk_text:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=metadata.copy()
                    ))
                current_pos = pos
        
        # Add remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                chunks.append(Document(
                    page_content=remaining,
                    metadata=metadata.copy()
                ))
        
        return chunks if chunks else [Document(page_content=text, metadata=metadata)]


class ChunkingStrategyFactory:
    """Factory to create chunking strategies"""
    
    @staticmethod
    def create_chunker(strategy: str, **kwargs) -> BaseChunker:
        """
        Create a chunker based on strategy name.
        
        Args:
            strategy: One of ['fixed', 'semantic', 'proposition', 'agentic']
            **kwargs: Strategy-specific parameters
        """
        strategy = strategy.lower()
        
        if strategy == "fixed":
            return FixedChunker(
                chunk_size=kwargs.get("chunk_size", Config.CHUNK_SIZE),
                chunk_overlap=kwargs.get("chunk_overlap", Config.CHUNK_OVERLAP)
            )
        
        elif strategy == "semantic":
            return SemanticChunker(
                model_name=kwargs.get("model_name", Config.EMBEDDING_MODEL),
                similarity_threshold=kwargs.get("similarity_threshold", 0.7),
                max_chunk_size=kwargs.get("max_chunk_size", 1500)
            )
        
        elif strategy == "proposition":
            return PropositionChunker(
                max_propositions_per_chunk=kwargs.get("max_propositions", 5)
            )
        
        elif strategy == "agentic":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key required for agentic chunking")
            return AgenticChunker(
                api_key=api_key,
                max_chunk_size=kwargs.get("max_chunk_size", 1500)
            )
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
if __name__ == "__main__":
    documents = []
    chunker = ChunkingStrategyFactory.create_chunker("fixed", chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"Number of chunks: {len(chunks)}")