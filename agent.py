
from langchain_ollama import ChatOllama
# 1. Initialize Local LLM (Optimized for 8GB RAM)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

def generate_final_answer(question, ranked_docs):
    # Take the top 3 documents based on RRF score
    context = "\n\n".join([doc[0] for doc in ranked_docs[:3]])
    
    prompt = f"""You are a helpful assistant. Use the provided context to answer the question.
    If the context doesn't contain the answer, say you don't know.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    # Use your llama3.2:3b model
    response = llm.invoke(prompt)
    return response.content

# --- Day 2 Execution Flow ---
# 1. Rewrite Query -> 3 Queries
# 2. Retrieve Docs for each Query
# 3. Apply RRF Score to merge results
# 4. Generate Answer using top context