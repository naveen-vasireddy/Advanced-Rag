from langgraph.graph import StateGraph, START, END
from agent_state import AgentState
from rewriter_node import rewriter_node
from reranker_node import reranker_node
from answering_node import answering_node
from retriver_node import retriver_node


builder = StateGraph(AgentState)
# Add your specialized nodes
builder.add_node("rewriter", rewriter_node)
builder.add_node("retriever", retriver_node)
builder.add_node("reranker", reranker_node)
builder.add_node("answerer", answering_node)

# Define the flow: START -> Rewriter -> Reranker -> Answerer -> END
builder.add_edge(START, "rewriter")
builder.add_edge("rewriter", "retriever")
builder.add_edge("retriever", "reranker")
builder.add_edge("reranker", "answerer")
builder.add_edge("answerer", END)

# Compile the Agent
rag_v2_agent = builder.compile()