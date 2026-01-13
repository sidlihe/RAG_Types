"""
Advanced RAG System - Gradio UI

Features:
- Multiple chunking strategy selection
- Multiple retrieval strategy selection
- Real-time metrics display
- Evaluation dashboard
- Source citations
- Strategy comparison
"""

import gradio as gr
import os
import shutil
import json
from datetime import datetime
from src.ingestion import load_and_split_pdf
from src.rag_engine import RAGPipeline
from src.utils import get_logger
from src.config import Config

logger = get_logger("App")

# Global State
pipeline_instance = None
current_chunks = None
conversation_history = []


def process_file(file_obj, api_key_input, chunking_strategy, chunk_size, chunk_overlap):
    """Process uploaded PDF with selected chunking strategy"""
    global pipeline_instance, current_chunks
    
    if not file_obj:
        return "⚠️ Please upload a PDF file.", ""
    
    # API Key Handling
    final_api_key = api_key_input.strip() if api_key_input else os.getenv("GOOGLE_API_KEY")
    
    if not final_api_key:
        return "❌ Error: No API Key provided (Input or .env).", ""
    
    try:
        # Initialize Pipeline
        pipeline_instance = RAGPipeline(api_key=final_api_key)
        
        # Save file
        if not os.path.exists("data"):
            os.makedirs("data")
        
        local_path = os.path.join("data", os.path.basename(file_obj.name))
        shutil.copy(file_obj.name, local_path)
        
        # Process with selected strategy
        status_msg = f"⚙️ Processing with {chunking_strategy} chunking..."
        
        kwargs = {}
        if chunking_strategy == "fixed":
            kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        
        current_chunks = load_and_split_pdf(
            local_path,
            chunking_strategy=chunking_strategy,
            api_key=final_api_key,
            **kwargs
        )
        
        # Ingest
        pipeline_instance.ingest_documents(current_chunks)
        
        # Generate chunk info
        chunk_info = f"""📊 **Chunking Summary:**
- Strategy: {chunking_strategy}
- Total Chunks: {len(current_chunks)}
- Avg Chunk Size: {sum(len(c.page_content) for c in current_chunks) // len(current_chunks)} chars
- Document: {os.path.basename(local_path)}
"""
        
        return f"✅ Document processed successfully!", chunk_info
        
    except Exception as e:
        logger.error(f"UI Error: {e}")
        return f"❌ Error: {str(e)}", ""


def answer_question(message, history, retrieval_strategy, enable_eval):
    """Answer question with selected retrieval strategy"""
    global pipeline_instance, conversation_history
    
    if pipeline_instance is None:
        return "⚠️ Please upload and process a PDF document first."
    
    try:
        # Query with options
        result = pipeline_instance.query(
            question=message,
            strategy=retrieval_strategy,
            enable_optimization=True,
            enable_evaluation=enable_eval
        )
        
        answer = result["answer"]
        
        # Add metadata footer
        metadata_lines = [
            f"\n\n---",
            f"📊 **Query Info:**",
            f"- Strategy: {result['strategy_used']}",
            f"- Latency: {result['latency_ms']:.0f}ms",
            f"- Contexts Retrieved: {result['num_contexts']}"
        ]
        
        if result.get("query_type"):
            metadata_lines.append(f"- Query Type: {result['query_type']}")
        
        # Add evaluation metrics if available
        if enable_eval and "evaluation" in result:
            eval_data = result["evaluation"]
            metadata_lines.append(f"\n📈 **Evaluation:**")
            
            if eval_data.get("faithfulness"):
                metadata_lines.append(f"- Faithfulness: {eval_data['faithfulness']:.3f}")
            if eval_data.get("answer_relevancy"):
                metadata_lines.append(f"- Answer Relevancy: {eval_data['answer_relevancy']:.3f}")
        
        # Add sources
        if Config.SHOW_SOURCE_CITATIONS and result.get("sources"):
            metadata_lines.append(f"\n📚 **Sources:**")
            for i, source in enumerate(result["sources"][:3], 1):
                metadata_lines.append(f"{i}. {source['source_file']} (Page {source['page']})")
        
        full_answer = answer + "\n".join(metadata_lines)
        
        # Save to history
        conversation_history.append({
            "question": message,
            "answer": answer,
            "metadata": result
        })
        
        return full_answer
        
    except Exception as e:
        logger.error(f"Query Error: {e}")
        return f"❌ Error: {str(e)}"


def get_metrics_dashboard():
    """Generate metrics dashboard"""
    global pipeline_instance
    
    if not pipeline_instance:
        return "No metrics available. Process a document and ask questions first."
    
    summary = pipeline_instance.get_metrics_summary()
    
    if "message" in summary:
        return summary["message"]
    
    # Format metrics
    lines = ["# 📊 Performance Metrics Dashboard\n"]
    
    for metric_name, stats in summary.items():
        lines.append(f"## {metric_name.replace('_', ' ').title()}")
        lines.append(f"- Mean: {stats['mean']:.3f}")
        lines.append(f"- Std Dev: {stats['std']:.3f}")
        lines.append(f"- Min: {stats['min']:.3f}")
        lines.append(f"- Max: {stats['max']:.3f}")
        lines.append(f"- Count: {stats['count']}\n")
    
    return "\n".join(lines)


def compare_strategies_ui(question, strategies_selected):
    """Compare multiple strategies"""
    global pipeline_instance
    
    if not pipeline_instance:
        return "⚠️ Please upload and process a document first."
    
    if not strategies_selected:
        return "⚠️ Please select at least one strategy to compare."
    
    try:
        comparison = pipeline_instance.compare_strategies(
            question=question,
            strategies=strategies_selected
        )
        
        return comparison["summary"]
        
    except Exception as e:
        logger.error(f"Comparison Error: {e}")
        return f"❌ Error: {str(e)}"


def export_conversation():
    """Export conversation history"""
    global conversation_history
    
    if not conversation_history:
        return None, "No conversation to export."
    
    # Create export
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "conversation": conversation_history
    }
    
    filepath = f"conversation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return filepath, f"✅ Exported to {filepath}"


# === Gradio UI ===
with gr.Blocks(title="Advanced RAG System") as demo:
    gr.Markdown("# 🚀 Advanced RAG System with Multi-Strategy Support")
    gr.Markdown("Upload documents, select chunking & retrieval strategies, and get AI-powered answers with comprehensive evaluation.")
    
    with gr.Tabs():
        # Tab 1: Document Processing
        with gr.Tab("📄 Document Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    api_key_input = gr.Textbox(
                        label="🔑 Gemini API Key",
                        placeholder="Paste key here or leave empty to use .env",
                        type="password"
                    )
                    
                    file_input = gr.File(
                        label="📁 Upload PDF",
                        file_types=[".pdf"]
                    )
                    
                    chunking_strategy = gr.Dropdown(
                        label="🧩 Chunking Strategy",
                        choices=["fixed", "semantic", "proposition", "agentic"],
                        value="fixed",
                        info="Select how to split the document"
                    )
                    
                    with gr.Accordion("Advanced Chunking Options", open=False):
                        chunk_size = gr.Slider(
                            label="Chunk Size",
                            minimum=200,
                            maximum=2000,
                            value=1000,
                            step=100
                        )
                        chunk_overlap = gr.Slider(
                            label="Chunk Overlap",
                            minimum=0,
                            maximum=500,
                            value=200,
                            step=50
                        )
                    
                    process_btn = gr.Button("⚙️ Process Document", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Status", interactive=False)
                    chunk_info = gr.Markdown("Chunk information will appear here...")
            
            process_btn.click(
                fn=process_file,
                inputs=[file_input, api_key_input, chunking_strategy, chunk_size, chunk_overlap],
                outputs=[status_output, chunk_info]
            )
        
        # Tab 2: Chat Interface
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.ChatInterface(
                        fn=answer_question,
                        additional_inputs=[
                            gr.Dropdown(
                                label="🔍 Retrieval Strategy",
                                choices=["dense", "sparse", "hybrid", "hybrid_rerank", 
                                        "parent_child", "multi_query", "hyde"],
                                value="hybrid_rerank",
                                info="Strategy for retrieving relevant chunks"
                            ),
                            gr.Checkbox(
                                label="📊 Enable Evaluation",
                                value=False,
                                info="Calculate RAGAS metrics (slower)"
                            )
                        ],
                        examples=[
                            ["What is the main summary?", "hybrid_rerank", False]
                            # ["List the key skills mentioned.", "hybrid_rerank", False],
                            # ["What is the educational background?", "hybrid_rerank", False],
                            # ["Summarize the work experience.", "hybrid_rerank", False]
                        ],
                        title="Ask Questions About Your Document"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Strategy Guide")
                    gr.Markdown("""
**Retrieval Strategies:**

- **Dense**: Pure semantic search
- **Sparse**: Keyword-based (BM25)
- **Hybrid**: Best of both worlds
- **Hybrid + Rerank**: Most accurate (recommended)
- **Parent-Child**: Precise retrieval with context
- **Multi-Query**: Multiple query variations
- **HyDE**: Hypothetical document embeddings

**When to use:**
- Factual questions → Hybrid Rerank
- Complex analysis → HyDE
- Summaries → Parent-Child
- Comparisons → Multi-Query
                    """)
        
        # Tab 3: Evaluation Dashboard
        with gr.Tab("📊 Metrics Dashboard"):
            gr.Markdown("## Performance Metrics")
            gr.Markdown("View aggregated metrics from your queries")
            
            refresh_btn = gr.Button("🔄 Refresh Metrics")
            metrics_display = gr.Markdown("Click refresh to load metrics...")
            
            refresh_btn.click(
                fn=get_metrics_dashboard,
                outputs=metrics_display
            )
            
            gr.Markdown("---")
            gr.Markdown("## Export Conversation")
            
            export_btn = gr.Button("💾 Export Conversation History")
            export_file = gr.File(label="Download")
            export_status = gr.Textbox(label="Export Status")
            
            export_btn.click(
                fn=export_conversation,
                outputs=[export_file, export_status]
            )
        
        # Tab 4: Strategy Comparison
        with gr.Tab("⚖️ Compare Strategies"):
            gr.Markdown("## Compare Multiple Retrieval Strategies")
            gr.Markdown("Test the same question with different strategies to see which performs best")
            
            with gr.Row():
                with gr.Column():
                    compare_question = gr.Textbox(
                        label="Question to Test",
                        placeholder="Enter a question to compare across strategies..."
                    )
                    
                    strategies_to_compare = gr.CheckboxGroup(
                        label="Select Strategies to Compare",
                        choices=["dense", "sparse", "hybrid", "hybrid_rerank", 
                                "parent_child", "multi_query", "hyde"],
                        value=["hybrid", "hybrid_rerank", "hyde"]
                    )
                    
                    compare_btn = gr.Button("🔬 Run Comparison", variant="primary")
                
                with gr.Column():
                    comparison_results = gr.Markdown("Comparison results will appear here...")
            
            compare_btn.click(
                fn=compare_strategies_ui,
                inputs=[compare_question, strategies_to_compare],
                outputs=comparison_results
            )
        
        # Tab 5: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
# Advanced RAG System

## Features

### 🧩 Chunking Strategies
- **Fixed**: Traditional recursive character splitting
- **Semantic**: Groups sentences by semantic similarity
- **Proposition**: Splits into atomic facts/statements
- **Agentic**: Uses LLM to intelligently determine boundaries

### 🔍 Retrieval Strategies
- **Dense**: Pure semantic search using embeddings
- **Sparse**: BM25 keyword-based search
- **Hybrid**: Ensemble of dense + sparse
- **Hybrid + Rerank**: Hybrid with cross-encoder reranking (most accurate)
- **Parent-Child**: Retrieve small chunks, return parent context
- **Multi-Query**: Generate multiple query variations
- **HyDE**: Generate hypothetical documents for better retrieval

### 📊 Evaluation Metrics
- **RAGAS**: Faithfulness, Answer Relevancy, Context Precision
- **METEOR**: Semantic similarity scoring
- **ROUGE**: N-gram overlap metrics
- **Retrieval Metrics**: MRR, NDCG, Precision@K

### 🎯 Query Optimization
- Automatic query classification
- Query expansion with synonyms
- Query rewriting for clarity
- Intent detection
- Automatic strategy selection

## Architecture

Built with:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- Gemini 2.0 Flash for LLM
- HuggingFace for embeddings & reranking
- RAGAS for evaluation
- Gradio for UI

## Credits

Developed as an advanced RAG demonstration system.
            """)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)