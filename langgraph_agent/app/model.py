import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# Get the base URL from the environment, defaulting to localhost if not set
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"🔗 Connecting to Ollama at: {ollama_url}")

# Pass the base_url to both the LLM and Embeddings
llm = ChatOllama(
    model="llama3.2:3b", 
    base_url=ollama_url, 
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="all-minilm", 
    base_url=ollama_url
)