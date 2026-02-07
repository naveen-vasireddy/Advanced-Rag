import os
from dotenv import load_dotenv
from langchain_classic.schema import Document
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import ChatOllama,OllamaEmbeddings
from model import llm, embeddings

load_dotenv()

def main():
    print("🚀 Setting up Day 53: Self-Querying Retriever...")

    # --- 1. Create Mock Data with METADATA ---
    # The "Metadata" is the structured data we want to filter by.
    docs = [
        Document(
            page_content="The Q3 server budget was $50,000.",
            metadata={"year": 2023, "department": "IT"}
        ),
        Document(
            page_content="We plan to hire 5 new engineers.",
            metadata={"year": 2023, "department": "HR"}
        ),
        Document(
            page_content="The server maintenance cost increased by 20%.",
            metadata={"year": 2024, "department": "IT"}
        ),
        Document(
            page_content="The company picnic is scheduled for July.",
            metadata={"year": 2024, "department": "HR"}
        ),
    ]

    # --- 2. Initialize Vector DB ---
    vectorstore = Chroma.from_documents(docs, embeddings)

    # --- 3. Define Metadata Schema ---
    # We must tell the LLM what fields exist so it knows how to write the filter.
    metadata_field_info = [
        AttributeInfo(
            name="year",
            description="The year the document relates to (integer)",
            type="integer",
        ),
        AttributeInfo(
            name="department",
            description="The department: 'IT' or 'HR'",
            type="string",
        ),
    ]
    document_content_description = "Internal company memos"

    # --- 4. Initialize the Self-Querying Retriever ---
    # This uses an LLM (gpt-4o or similar) to translate text -> metadata filters
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True # Set to True to see the internal filter generation
    )

    # --- 5. Test Case ---
    # Notice: This query is vague semantically ("server costs") but specific on metadata ("2024").
    # A standard vector search might return 2023 data because it matches "server".
    # This retriever should EXCLUDE 2023 data.
    query = "Tell me about server costs in 2024."
    
    print(f"\n🔍 Query: '{query}'")
    results = retriever.invoke(query)

    print("\n🏆 Results (Check the Metadata year!):")
    for res in results:
        print(f"   - Content: {res.page_content}")
        print(f"     Metadata: {res.metadata}")

if __name__ == "__main__":
    main()