import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import re
# 1. Initialize Local LLM (Optimized for 8GB RAM)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# 2. Setup Query Re-Writer Chain
# We explicitly ask for 3 variations: 1 Question, 1 Concept, 1 Keyword
template = """You are an AI Search Optimizer. The user is asking a question about local AI or hardware.
Your goal is to generate 3 specific search queries to find the best information in a technical database.

1. First, a direct natural language question.
2. Second, a technical keyword-based query.
3. Third, a conceptual search query that covers broader context.

Original Question: {question}

Provide ONLY the 3 queries, one per line, no headers or numbers."""

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