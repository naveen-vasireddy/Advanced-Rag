
from agent_state import AgentState
from model import llm
from retry_func import node_with_retry

def answering_node(state: AgentState):
    def process(s):
        if not s.get("reranked_docs"):
            return {"final_answer": "I'm sorry, I couldn't find any relevant information to answer your question."}
        
        # Existing logic [6, 7]
        question = s["question"]
        context = "\n\n".join([doc for doc in s["reranked_docs"]])
        prompt = f"""You are a helpful assistant. Use the provided context to answer the question.
            If the context doesn't contain the answer, say you don't know.
            
            Context:
            {context}
            
            Question: {question}
            Answer:"""        
        response = llm.invoke(prompt)
        return {"final_answer": response.content}

    return node_with_retry(process, state, "Answerer")
