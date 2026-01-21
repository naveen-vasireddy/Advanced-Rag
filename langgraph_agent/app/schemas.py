from typing import List, Optional
from pydantic import BaseModel, Field, validator

class QueryExpansion(BaseModel):
    """Schema for the Query Re-Writer output."""
    natural_language: str = Field(
        description="A direct, natural language version of the user query."
    )
    keyword_search: str = Field(
        description="A boolean or keyword-focused search query for technical databases."
    )
    conceptual_search: str = Field(
        description="A broad, thematic search query to capture context."
    )

    # Validator to ensure queries aren't empty or lazy
    @validator('natural_language', 'keyword_search', 'conceptual_search')
    def check_length(cls, v):
        if len(v) < 5:
            raise ValueError("Query is too short to be effective.")
        return v
    

class RetrievalInput(BaseModel):
    query: str = Field(..., description="The specific query string to search in the vector DB.")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of documents to retrieve.")
    
    @validator("query")
    def clean_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query is too short for semantic search.")
        return v.strip()

class FinalResponse(BaseModel):
    answer: str = Field(..., description="The synthesized answer to the user.")
    citations: List[str] = Field(default_factory=list, description="Direct quotes or document names used.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence (0.0-1.0) based on context availability.")