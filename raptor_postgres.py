import os
import numpy as np
from dotenv import load_dotenv
from typing import List

# Database & AI
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Clustering
from sklearn.cluster import KMeans

load_dotenv()

# --- CONFIGURATION ---
CONNECTION_STRING = "postgresql+psycopg2://user:password@localhost:5432/vectordb"
COLLECTION_NAME = "my_documents" # The table name in Postgres
NUM_CLUSTERS = 3  # How many "topics" you want to find

def main():
    print("🚀 Starting RAPTOR-Lite on Postgres...")

    # 1. Initialize Database & Embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    # 2. Fetch All Vectors (for Clustering)
    # PGVector doesn't have a simple "get_all" method, so we usually expose the underlying table
    # Or purely for this demo, we assume we have documents loaded. 
    # Here is a helper to fetch them via pure SQL or LangChain trickery.
    # For safety, we will assume you have a list of docs. In prod, query the table directly.
    print("📥 Fetching vectors from Postgres...")
    
    # NOTE: In a real PGVector setup, you would run: "SELECT embedding, document FROM langchain_pg_embedding"
    # For this script to be runnable, let's pretend we just loaded these docs:
    mock_docs = [
        "The server CPU load is high.", "Memory usage is at 90%.", # Cluster 1: Hardware
        "The PTO policy allows 20 days.", "HR requires a sick note.", # Cluster 2: HR
        "Q3 revenue was $1M.", "Profit margins are up by 5%." # Cluster 3: Finance
    ]
    docs = [Document(page_content=t) for t in mock_docs]
    vectorstore.add_documents(docs) # Ensure they are in DB
    
    # Retrieve embeddings for clustering
    vectors = embeddings.embed_documents([d.page_content for d in docs])
    matrix = np.array(vectors)

    # --- PART 1: K-MEANS CLUSTERING ---
    print(f"🧮 Running K-Means (k={NUM_CLUSTERS})...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    # Group documents by their Cluster ID
    clustered_docs = {i: [] for i in range(NUM_CLUSTERS)}
    for idx, label in enumerate(labels):
        clustered_docs[label].append(docs[idx])

    # --- PART 2: RAPTOR SUMMARIZATION ---
    print("📝 Generating Cluster Summaries (RAPTOR Step)...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    summary_prompt = ChatPromptTemplate.from_template(
        "Here are documents from a specific topic cluster:\n\n{context}\n\n"
        "Summarize the main theme and key details of this cluster in 2 sentences."
    )
    chain = summary_prompt | llm | StrOutputParser()

    cluster_summaries = []
    
    for i in range(NUM_CLUSTERS):
        # Join all text in this cluster
        cluster_text = "\n".join([d.page_content for d in clustered_docs[i]])
        
        # Summarize the cluster
        summary = chain.invoke({"context": cluster_text})
        cluster_summaries.append(f"Cluster {i+1} Summary: {summary}")
        print(f"   > Topic {i+1}: {summary}")

    # --- PART 3: GLOBAL ANSWERING ---
    # Now we answer a "Global" question using ONLY the summaries.
    # Standard RAG would fail here because it would only fetch 2-3 random specific docs.
    
    global_query = "Give me a high-level overview of everything going on in the company."
    print(f"\n🌍 Global Query: '{global_query}'")

    final_prompt = ChatPromptTemplate.from_template(
        "You are a strategic assistant. Answer the user query based on these topic summaries:\n\n"
        "{summaries}\n\n"
        "User Query: {query}"
    )
    
    final_chain = final_prompt | llm | StrOutputParser()
    
    response = final_chain.invoke({
        "summaries": "\n\n".join(cluster_summaries),
        "query": global_query
    })

    print(f"\n🏆 RAPTOR Response:\n{response}")

if __name__ == "__main__":
    main()