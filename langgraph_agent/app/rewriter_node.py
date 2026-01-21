
from langgraph_agent.app.model import llm
from schemas import QueryExpansion
from decorators import validate_llm_output

# 1. Apply the validation decorator
@validate_llm_output(QueryExpansion)
def generate_queries_chain(state):
    question = state["question"]
    
    # 2. Update Prompt to strictly request JSON
    prompt = f"""You are an AI Search Optimizer. The user is asking a question about local AI or hardware.
        Your goal is to generate 3 specific search queries to find the best information in a technical database.

        1. First, a direct natural language question.
        2. Second, a technical keyword-based query.
        3. Third, a conceptual search query that covers broader context.

        Original Question: {question}
    
    You MUST return a JSON object with the following keys:
    - "natural_language"
    - "keyword_search"
    - "conceptual_search"
    
    Do not add any conversational text. Return ONLY JSON."""
    response = llm.invoke(prompt).content
    return response


def rewriter_node(state):
    print(f"Rewriting: {state['question']}")
    try:
        # 3. Call the decorated function
        query_obj = generate_queries_chain(state)
        
        # 4. Convert Pydantic object back to list for the State
        queries = [
            query_obj.natural_language,
            query_obj.keyword_search,
            query_obj.conceptual_search
        ]
        return {"rewritten_queries": queries}
        
    except Exception as e:
        print(f"Fallback due to error: {e}")
        # Fail-safe: just search the original question
        return {"rewritten_queries": [state["question"]]}