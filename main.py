from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import re

# 1. Initialize Local LLM (Optimized for 8GB RAM)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# 2. Setup Query Re-Writer Chain
# We explicitly ask for 3 variations: 1 Question, 1 Concept, 1 Keyword
template = """You are an AI assistant. Your task is to generate 
THREE different versions of the user question to help retrieve 
documents from a vector database. 

Structure your response as:
1. A natural language question
2. A conceptual variation
3. A keyword-based search query

Original question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

def parse_queries(llm_output):
    # Split by lines
    lines = llm_output.strip().split("\n")
    
    clean_queries = []
    for line in lines:
        # Remove numbering (e.g., "1. ", "1) ") and bullet points
        cleaned = re.sub(r'^\d+[\.\)]\s*|^\*\s*', '', line).strip()
        
        # Ignore lines that are too short or look like headers/filler
        if (len(cleaned) > 10 and 
            not cleaned.lower().startswith("here are") and 
            not cleaned.lower().startswith("original question") and
            "**" not in cleaned):
            clean_queries.append(cleaned)
    
    # Return exactly the first 3 valid queries found
    return clean_queries[:3]

# 2. Update your chain
generate_queries = (
    prompt 
    | llm 
    | StrOutputParser() 
    | parse_queries  # Using our new robust parser
)

# 3. Setup Lightweight Embeddings
# 'all-minilm' is much faster and uses less RAM than 'nomic'
embeddings = OllamaEmbeddings(model="all-minilm")

# Sample Data to test
docs = [
    Document(page_content="The AMD Ryzen 5 4600H is a 6-core processor designed for laptops."),
    Document(page_content="8GB of RAM is the minimum recommended for running local LLMs comfortably."),
    Document(page_content="Using small models like Llama 3.2 3B helps avoid system memory pressure.")
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def get_documents_for_multiple_queries(queries):
    print(f"\n🚀 System is using {len(queries)} specific queries:")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")
        
    all_docs = []
    for q in queries:
        all_docs.append(retriever.invoke(q)) # Keep as lists of lists for RRF tomorrow
    
    return all_docs


# 5. Execute
question = "How much RAM do I need for local AI?"
final_chain = generate_queries | get_documents_for_multiple_queries

results = final_chain.invoke({"question": question})

print("--- Retrieved Documents ---")
for doc in results:
    print(f"- {doc[0].page_content}")