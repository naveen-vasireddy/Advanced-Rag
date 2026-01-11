
from agent_state import AgentState
from model import llm

def answering_node(state: AgentState):
    print("Generating final answer using retrieved documents.")
    question = state["question"]
    context = "\n\n".join([doc for doc in state["reranked_docs"]])
        
    prompt = f"""You are a helpful assistant. Use the provided context to answer the question.
    If the context doesn't contain the answer, say you don't know.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    # Use your llama3.2:3b model
    response = llm.invoke(prompt)
    return {"final_answer": response.content}