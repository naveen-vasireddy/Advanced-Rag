import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.schema import Document

load_dotenv()

def main():
    print("🚀 Setting up Hybrid RAG System...")

    # --- 1. Create Mock Data ---
    # Notice the specific acronyms and IDs (ISO-9001, Project Alpha) 
    # that vector search sometimes "hallucinates" or misses.
    docs_text = [
        "The return policy for items under $50 is 30 days.",
        "For defective electronics, referencing ISO-9001 standards is required.",
        "Standard shipping takes 3-5 business days via Ground transport.",
        "The project code is 'Project Alpha' for all internal memos.",
        "Apples are red and bananas are yellow fruit."
    ]
    documents = [Document(page_content=t) for t in docs_text]

    # --- 2. Initialize Sparse Retriever (Keyword / BM25) ---
    # This acts like a standard search engine (Ctrl+F style logic)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2
    print("✅ BM25 (Keyword) Retriever ready.")

    # --- 3. Initialize Dense Retriever (Vector / Semantic) ---
    # We use a temporary in-memory Chroma DB
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = Chroma.from_documents(documents, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("✅ Vector (Semantic) Retriever ready.")

    # --- 4. Initialize Hybrid Retriever (Ensemble) ---
    # We weight them 50/50. The EnsembleRetriever uses "Reciprocal Rank Fusion"
    # to re-order results from both sources.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # You can adjust weights here
    )

    # --- 5. The Test ---
    # A query that requires EXACT matching of an acronym ("ISO-9001")
    query = "What standards apply to defective electronics (ISO-9001)?"
    
    print(f"\n🔍 Querying: '{query}'")
    
    # Run the hybrid search
    results = ensemble_retriever.invoke(query)

    print("\n🏆 Top Hybrid Results:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content}")

if __name__ == "__main__":
    main()