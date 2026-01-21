from langgraph_agent.app.agent_graph import rag_v2_agent
from langgraph_agent.app.agent_state import AgentState
import os
import dotenv
dotenv.load_dotenv()
# Essential for Phase 2 tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG-v2-Optimization" # Name your project

if __name__ == "__main__":
    # Example question
    question = "Local LLM hardware requirements and CPU inference"
    
    # Initialize the agent state
    initial_state: AgentState = {
        "question": question,
        "rewritten_queries": [],
        "retrieved_docs": [],
        "final_answer": ""
    }
    
    # Run the RAG v2 Agent
    final_state = rag_v2_agent.invoke(initial_state)
    
    # Print the final answer
    print("Final Answer:", final_state["final_answer"])