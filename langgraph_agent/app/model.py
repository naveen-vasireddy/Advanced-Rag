import os
from langchain_openai import ChatOpenAI     
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv  # <-- Add this
load_dotenv() 

# 1. Setup the LLM (DeepSeek R1T2 Chimera via OpenRouter)
# Ensure you have set OPENROUTER_API_KEY in your environment variables
llm = ChatOpenAI(
    model="tngtech/deepseek-r1t2-chimera:free",      # The specific free model ID
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),   # Your OpenRouter Key
    openai_api_base="https://openrouter.ai/api/v1",   # OpenRouter Base URL
    temperature=0.1                                   # Low temp is best for RAG/QnA
)

# 2. Setup Embeddings 
# Since you aren't using Ollama, we use a free local model (Standard for RAG)
# This downloads a small model to run locally without an API.
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small", 
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False # Optional: bypasses local token counting
)


