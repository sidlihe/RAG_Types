"""
Query Optimization Module

Implements intelligent query processing:
1. Query Classification - Categorize query type
2. Query Expansion - Add synonyms and related terms
3. Query Rewriting - Improve clarity and specificity
4. Intent Detection - Understand user intent
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils import get_logger

logger = get_logger("QueryOptimizer")


class QueryType(Enum):
    """Types of queries"""
    FACTUAL = "factual"  # Who, what, when, where
    ANALYTICAL = "analytical"  # Why, how, explain
    SUMMARIZATION = "summarization"  # Summarize, overview
    COMPARISON = "comparison"  # Compare, difference
    PROCEDURAL = "procedural"  # Steps, process, how-to
    OPINION = "opinion"  # Should, recommend


class QueryIntent(Enum):
    """User intent categories"""
    INFORMATION_SEEKING = "information_seeking"
    CLARIFICATION = "clarification"
    VERIFICATION = "verification"
    EXPLORATION = "exploration"


class QueryClassifier:
    """Classify queries into types"""
    
    def __init__(self):
        # Keywords for each query type
        self.type_keywords = {
            QueryType.FACTUAL: ["who", "what", "when", "where", "which", "name", "list"],
            QueryType.ANALYTICAL: ["why", "how", "explain", "analyze", "reason", "cause"],
            QueryType.SUMMARIZATION: ["summarize", "summary", "overview", "brief", "main points"],
            QueryType.COMPARISON: ["compare", "difference", "versus", "vs", "similar", "contrast"],
            QueryType.PROCEDURAL: ["steps", "process", "procedure", "how to", "guide"],
            QueryType.OPINION: ["should", "recommend", "suggest", "best", "opinion"]
        }
    
    def classify(self, query: str) -> QueryType:
        """
        Classify query type based on keywords.
        
        Args:
            query: User query
        
        Returns:
            QueryType enum
        """
        query_lower = query.lower()
        
        # Count matches for each type
        scores = {}
        for qtype, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[qtype] = score
        
        # Return type with highest score, default to FACTUAL
        if max(scores.values()) > 0:
            classified_type = max(scores, key=scores.get)
            logger.info(f"Query classified as: {classified_type.value}")
            return classified_type
        
        return QueryType.FACTUAL
    
    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect user intent.
        
        Args:
            query: User query
        
        Returns:
            QueryIntent enum
        """
        query_lower = query.lower()
        
        # Simple heuristics
        if any(word in query_lower for word in ["is it true", "correct", "verify", "confirm"]):
            return QueryIntent.VERIFICATION
        elif any(word in query_lower for word in ["clarify", "mean", "unclear", "confused"]):
            return QueryIntent.CLARIFICATION
        elif any(word in query_lower for word in ["explore", "tell me more", "what else", "related"]):
            return QueryIntent.EXPLORATION
        else:
            return QueryIntent.INFORMATION_SEEKING


class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=api_key
            )
        else:
            self.llm = None
    
    def expand_simple(self, query: str) -> List[str]:
        """
        Simple query expansion using basic synonyms.
        
        Args:
            query: Original query
        
        Returns:
            List of expanded query variations
        """
        # Basic synonym mapping
        synonyms = {
            "skills": ["abilities", "competencies", "expertise"],
            "experience": ["background", "work history", "career"],
            "education": ["academic background", "qualifications", "degrees"],
            "summary": ["overview", "synopsis", "brief"],
            "main": ["primary", "key", "principal", "important"]
        }
        
        expanded = [query]
        query_lower = query.lower()
        
        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns:
                    expanded_query = query_lower.replace(word, syn)
                    expanded.append(expanded_query)
        
        logger.info(f"Simple expansion: {len(expanded)} variations")
        return expanded[:5]  # Limit to 5 variations
    
    def expand_llm(self, query: str, num_variations: int = 3) -> List[str]:
        """
        LLM-based query expansion.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
        
        Returns:
            List of expanded query variations
        """
        if not self.llm:
            logger.warning("LLM not available for expansion, using simple expansion")
            return self.expand_simple(query)
        
        prompt = f"""Given the query: "{query}"

Generate {num_variations} alternative phrasings that preserve the original meaning but use different words.
Return ONLY the alternative queries, one per line, without numbering or explanations.

Alternative queries:"""
        
        try:
            response = self.llm.invoke(prompt)
            variations = [line.strip() for line in response.content.split('\n') if line.strip()]
            variations = [query] + variations[:num_variations]
            
            logger.info(f"LLM expansion: {len(variations)} variations")
            return variations
        
        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return self.expand_simple(query)


class QueryRewriter:
    """Rewrite queries for better clarity and retrieval"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            google_api_key=api_key
        )
    
    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite query for better clarity.
        
        Args:
            query: Original query
            context: Optional conversation context
        
        Returns:
            Rewritten query
        """
        context_str = f"\nConversation context: {context}" if context else ""
        
        prompt = f"""Rewrite the following query to be more clear, specific, and optimized for document retrieval.
Keep the core meaning but make it more explicit.{context_str}

Original query: "{query}"

Rewritten query:"""
        
        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        
        Args:
            query: Complex query
        
        Returns:
            List of simpler sub-queries
        """
        prompt = f"""Break down the following complex query into simpler, atomic sub-queries.
Each sub-query should address one specific aspect.

Complex query: "{query}"

Sub-queries (one per line):"""
        
        try:
            response = self.llm.invoke(prompt)
            sub_queries = [line.strip() for line in response.content.split('\n') if line.strip()]
            
            logger.info(f"Query decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
        
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]


class QueryOptimizer:
    """Main query optimization orchestrator"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.classifier = QueryClassifier()
        self.expander = QueryExpander(api_key)
        self.rewriter = QueryRewriter(api_key) if api_key else None
    
    def optimize(self, 
                query: str,
                enable_classification: bool = True,
                enable_expansion: bool = False,
                enable_rewriting: bool = False) -> Dict[str, any]:
        """
        Optimize query with selected techniques.
        
        Args:
            query: Original query
            enable_classification: Classify query type
            enable_expansion: Expand query with variations
            enable_rewriting: Rewrite query for clarity
        
        Returns:
            Dictionary with optimization results
        """
        result = {
            "original_query": query,
            "optimized_query": query,
            "query_type": None,
            "intent": None,
            "expansions": [],
            "sub_queries": []
        }
        
        # Classification
        if enable_classification:
            result["query_type"] = self.classifier.classify(query)
            result["intent"] = self.classifier.detect_intent(query)
        
        # Expansion
        if enable_expansion:
            if self.expander.llm:
                result["expansions"] = self.expander.expand_llm(query)
            else:
                result["expansions"] = self.expander.expand_simple(query)
        
        # Rewriting
        if enable_rewriting and self.rewriter:
            result["optimized_query"] = self.rewriter.rewrite(query)
            
            # Check if query is complex
            if len(query.split()) > 15 or "and" in query.lower():
                result["sub_queries"] = self.rewriter.decompose_complex_query(query)
        
        logger.info(f"Query optimization complete: type={result.get('query_type')}, "
                   f"intent={result.get('intent')}")
        
        return result
    
    def get_retrieval_strategy_recommendation(self, query_type: QueryType) -> str:
        """
        Recommend retrieval strategy based on query type.
        
        Args:
            query_type: Classified query type
        
        Returns:
            Recommended retrieval strategy name
        """
        recommendations = {
            QueryType.FACTUAL: "hybrid_rerank",  # Precision important
            QueryType.ANALYTICAL: "hyde",  # Complex reasoning
            QueryType.SUMMARIZATION: "parent_child",  # Need context
            QueryType.COMPARISON: "multi_query",  # Multiple perspectives
            QueryType.PROCEDURAL: "parent_child",  # Sequential context
            QueryType.OPINION: "hybrid_rerank"  # Balanced approach
        }
        
        strategy = recommendations.get(query_type, "hybrid_rerank")
        logger.info(f"Recommended strategy for {query_type.value}: {strategy}")
        return strategy
