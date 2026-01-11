from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

llm = ChatOllama(model="llama3.2:3b", temperature=0)

embeddings = OllamaEmbeddings(model="all-minilm")