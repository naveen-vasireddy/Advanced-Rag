# spec.py
from pydantic import BaseModel, Field
from typing import List, Optional

class AgentRequest(BaseModel):
    user_id: str = Field(..., description="Unique ID for user (used for memory)")
    session_id: str = Field(..., description="Conversation ID for LangGraph threads")
    query: str = Field(..., description="Raw input text (potentially containing PII)")

class AgentResponse(BaseModel):
    answer: str = Field(..., description="The final answer (Deanonymized)")
    redacted_entities: List[str] = Field(..., description="List of PII types found (e.g., 'EMAIL')")
    latency_ms: float = Field(..., description="Total execution time")
    cost_estimate: float = Field(..., description="Estimated cost of this turn")
