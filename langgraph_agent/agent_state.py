import operator
from typing import TypedDict, List, Annotated

class AgentState(TypedDict):
    question: str
    rewritten_queries: List[str]
    retrieved_docs: Annotated[List[str], operator.add] # Appends results
    reranked_docs: List[str]
    final_answer: str
# Initialize the Graph
