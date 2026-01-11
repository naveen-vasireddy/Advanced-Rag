from agent_state import AgentState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from model import llm
import re
def rewriter_node(state: AgentState):
    # Logic for Query Re-Writer Agent (Days 25-27)
    print("Rewriting queries for question:", state["question"])
    question = state["question"]
    template = """You are an AI Search Optimizer. The user is asking a question about local AI or hardware.
    Your goal is to generate 3 specific search queries to find the best information in a technical database.

    1. First, a direct natural language question.
    2. Second, a technical keyword-based query.
    3. Third, a conceptual search query that covers broader context.

    Original Question: {question}

    Provide ONLY the 3 queries, one per line, no headers or numbers."""

    prompt = ChatPromptTemplate.from_template(template)
    # Implementation: Use an LLM to generate 3 variations
    # Use the pipe operator for a standard chain
    rewriter_chain = prompt | llm | StrOutputParser()

    # 3. Execution and list transformation
    response = rewriter_chain.invoke({"question": question})

    new_queries = parse_queries(response)
    return {"rewritten_queries": new_queries}


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