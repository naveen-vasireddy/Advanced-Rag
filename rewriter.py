import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define the Multi-Query Prompt
# We ask for 3 versions to improve retrieval coverage
template = """You are an AI language model assistant. Your task is to generate 
three different versions of the given user question to retrieve relevant 
documents from a vector database. By generating multiple perspectives on 
the user question, your goal is to help the user overcome some of the 
limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines.
Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

# 3. Build the Chain
generate_queries = (
    prompt_perspectives 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n")) # Split string into a list of 3 queries
)

# Test it
question = "What are the impacts of climate change on the environment?"
queries = generate_queries.invoke({"question": question})
print(f"Generated Queries: {queries}")