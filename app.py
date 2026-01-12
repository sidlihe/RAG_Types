import gradio as gr
import os
import shutil
from src.ingestion import load_and_split_pdf
from src.rag_engine import RAGPipeline
from src.utils import get_logger

logger = get_logger("App")

# Global State to hold the pipeline instance
pipeline_instance = None
current_chain = None

def process_file(file_obj, api_key_input):
    global pipeline_instance, current_chain
    
    if not file_obj:
        return "Please upload a PDF file."
    
    # 1. API Key Handling
    # Use input key if provided, otherwise fallback to .env
    final_api_key = api_key_input.strip() if api_key_input else os.getenv("GOOGLE_API_KEY")
    
    if not final_api_key:
        return "Error: No API Key provided (Input or .env)."
        
    try:
        # Initialize Pipeline
        pipeline_instance = RAGPipeline(api_key=final_api_key)
        
        # Save file to data dir
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Copy file to local data folder for stability
        local_path = os.path.join("data", os.path.basename(file_obj.name))
        shutil.copy(file_obj.name, local_path)
        
        # Process Document
        status_msg = f"Processing {os.path.basename(local_path)}..."
        print(status_msg)
        
        # Ingest
        chunks = load_and_split_pdf(local_path)
        pipeline_instance.ingest_documents(chunks)
        
        # Build Chain
        current_chain = pipeline_instance.build_chain(chunks)
        
        return f"✅ Document '{os.path.basename(local_path)}' processed successfully! You can now ask questions."
        
    except Exception as e:
        logger.error(f"UI Error: {e}")
        return f"❌ Error: {str(e)}"

def answer_question(message, history):
    global pipeline_instance, current_chain
    
    if current_chain is None:
        return "⚠️ Please upload and process a PDF document first."
    
    response = pipeline_instance.query(current_chain, message)
    return response

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Professional Hybrid RAG with Reranking")
    gr.Markdown("Upload a PDF, enter your Gemini API Key (optional if in .env), and chat with your document.")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Gemini API Key", 
                placeholder="Paste key here or leave empty to use .env",
                type="password"
            )
            file_input = gr.File(
                label="Upload PDF", 
                file_types=[".pdf"]
            )
            process_btn = gr.Button("⚙️ Process Document", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=answer_question,
                examples=["What is the main summary?"],
            )

    # Button Logic
    process_btn.click(
        fn=process_file,
        inputs=[file_input, api_key_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())