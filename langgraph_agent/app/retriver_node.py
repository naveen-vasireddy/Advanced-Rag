from langgraph_agent.app.agent_state import AgentState
from langgraph_agent.app.model import embeddings

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from schemas import RetrievalInput
from pydantic import ValidationError

from langgraph_agent.app.retry_func import node_with_retry

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
    def process(s):
        queries = s["rewritten_queries"]
        all_docs = []

        for q in queries:
            try:
                # 1. Validate the input using Pydantic before usage
                # This ensures 'q' meets length/content requirements defined in schema
                validated_input = RetrievalInput(query=q, top_k=2)
                
                # 2. Use the validated query string
                docs = retriever.invoke(validated_input.query)
                all_docs.append(docs)
                
            except ValidationError as ve:
                print(f"⚠️ Skipping invalid query '{q}': {ve}")
                continue # Skip bad queries without crashing
            except Exception as e:
                print(f"Warning: Retrieval failed for query '{q}': {e}")
                continue
        
        if not all_docs:
            print("Fallback: Using original question")
            # Fallback also validated? Optional, but good practice.
            all_docs.append(retriever.invoke(s["question"]))
            
        return {"retrieved_docs": all_docs}

    return node_with_retry(process, state, "Retriever")