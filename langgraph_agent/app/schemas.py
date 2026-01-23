from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class QueryExpansion(BaseModel):
    """
    Schema for the Query Re-Writer output. 
    Used by the Query Transformation node (Roadmap Days 25-27).
    """
    natural_language: str = Field(
        description="A direct, natural language version of the user query, optimized for clarity."
    )
    keyword_search: str = Field(
        description="A specific, keyword-focused search query optimized for technical databases."
    )
    conceptual_search: str = Field(
        description="A broad, thematic search query designed to capture high-level context."
    )

    # Use Pydantic v2 `field_validator` for per-field validation
    @field_validator('natural_language', 'keyword_search', 'conceptual_search')
    def check_length(cls, v: str) -> str:
        # Lowered limit to 3 to allow acronyms like 'RAG', 'AWS'
        if not isinstance(v, str) or len(v.strip()) < 3:
            raise ValueError("Query is too short (must be at least 3 chars).")
        return v.strip()
class RetrievalInput(BaseModel):
    query: str = Field(..., description="The specific query string to search in the vector DB.")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of documents to retrieve.")
    
    @field_validator("query")
    def clean_query(cls, v: str) -> str:
        if not isinstance(v, str) or len(v.strip()) < 3:
            raise ValueError("Query is too short for semantic search.")
        return v.strip()

class FinalResponse(BaseModel):
    answer: str = Field(..., description="The synthesized answer to the user.")
    citations: List[str] = Field(default_factory=list, description="Direct quotes or document names used.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence (0.0-1.0) based on context availability.")

    @field_validator("confidence_score")
    def check_confidence(cls, v) -> float:
        try:
            v = float(v)
        except Exception:
            raise ValueError("confidence_score must be a number between 0.0 and 1.0")
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v