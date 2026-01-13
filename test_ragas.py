
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.evaluation import RAGEvaluator

load_dotenv()

def test_evaluation():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        return

    print("Initializing models...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
    
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    contexts = ["France is a country in Europe. Its capital city is Paris."]
    
    print("Running evaluation...")
    result = evaluator.evaluate_ragas(question, answer, contexts)
    
    print("\nEvaluation Results:")
    for metric, score in result.items():
        print(f"{metric}: {score}")

if __name__ == "__main__":
    test_evaluation()
