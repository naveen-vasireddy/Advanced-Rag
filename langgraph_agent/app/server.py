from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_graph import rag_v2_agent # Import your compiled graph [2]

app = FastAPI(title="RAG v2.0 Agent API")

# Define the Input Schema (API Request)
class QueryRequest(BaseModel):
    question: str

# Define the Output Schema (API Response)
class QueryResponse(BaseModel):
    final_answer: str
    citations: list[str] = [] # Optional, based on your Day 42 work

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # Initialize state as defined in your agent_state.py [3]
        initial_state = {"question": request.question}
        
        # Invoke the LangGraph agent
        result = rag_v2_agent.invoke(initial_state)
        
        # Return the structured response
        return QueryResponse(
            final_answer=result["final_answer"],
            citations=result.get("citations", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)