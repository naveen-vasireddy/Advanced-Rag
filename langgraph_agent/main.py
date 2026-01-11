from agent_graph import rag_v2_agent
from agent_state import AgentState

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