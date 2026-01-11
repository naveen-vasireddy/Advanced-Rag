from agent_state import AgentState
from model import embeddings

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Replace your 'docs' variable with this
docs = [
    Document(page_content="""The AMD Ryzen 5 4600H is a high-performance 6-core processor. 
    It handles multi-threaded AI tasks by distributing computation across its Zen 2 cores. 
    On 8GB RAM systems, CPU-based inference is slower than GPU, but more stable for small models."""),
    
    Document(page_content="""Llama 3.2 3B is a 'Small Language Model' (SLM). 
    A 3B parameter model in 4-bit quantization typically requires ~2.5GB of VRAM/RAM. 
    This makes it ideal for 8GB RAM laptops, leaving room for the OS and background tasks."""),
    
    Document(page_content="""Reciprocal Rank Fusion (RRF) works by combining rankings from multiple searches. 
    It doesn't care about raw similarity scores. Instead, it uses the formula 1/(rank + k). 
    This ensures that documents found by multiple queries are prioritized."""),
    
    Document(page_content="""When running local AI on 8GB RAM, memory pressure is the main bottleneck. 
    Closing background apps like Chrome and using lightweight embedding models like 'all-minilm' 
    is essential to prevent the system from using 'Swap memory' on the SSD.""")
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def retriver_node(state: AgentState):
    print("Retrieving documents for rewritten queries.")
    queries = state["rewritten_queries"]
    print(f"\n🚀 System is using {len(queries)} specific queries:")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")
        
    all_docs = []
    for q in queries:
        all_docs.append(retriever.invoke(q)) # Keep as lists of lists for RRF tomorrow

    return {"retrieved_docs": all_docs}