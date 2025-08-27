import os
from rag_langgraph import RAGLangGraph

def test_rag_system():
    """Test the RAG system with various questions."""
    
    # Make sure API keys are set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    if not os.environ.get("TAVILY_API_KEY"):
        print("Please set TAVILY_API_KEY environment variable")
        return

    # Initialize the RAG system
    print("Initializing RAG system...")
    rag = RAGLangGraph()
    
    # Test questions
    test_questions = [
        "What are the types of agent memory?",
        "Explain prompt engineering techniques",
        "What are adversarial attacks on LLMs?",
        "Who won the 2024 Olympics basketball final?",  # Should route to web search
        "What is chain-of-thought prompting?"
    ]
    
    print("Running test queries...")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        print("-" * 60)
        
        try:
            result = rag.run_query(question)
            answer = result.get("generation", "No answer generated")
            print(f"\n✅ Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
        
        print("=" * 80)

if __name__ == "__main__":
    test_rag_system()