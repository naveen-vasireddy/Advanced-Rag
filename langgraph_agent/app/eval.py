from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI # Or your preferred high-reasoning LLM
from langchain_core.prompts import ChatPromptTemplate
from app.model import llm

# Define the scoring structure for the judge
class RAGEvaluation(BaseModel):
    relevance_score: int = Field(description="Score 1-5: How well the context answers the question.")
    groundedness_score: int = Field(description="Score 1-5: Is the answer derived ONLY from the context?")
    reasoning: str = Field(description="Explanation for the assigned scores.")

# Initialize a high-reasoning model to act as the "Judge"
judge_llm = llm.with_structured_output(RAGEvaluation)
# 2. The Evaluation Prompt
# This prompt directs the LLM to act as an impartial evaluator for your specific RAG v2.0 output.
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert grader evaluating a RAG system.  "
               "Relevance: Does the context provide the necessary info? "
               "Groundedness: Does the answer avoid hallucinations not present in the context?"),
    ("human", """
    ### USER QUESTION
    {question}

    ### RETRIEVED CONTEXT
    {context}

    ### AGENT ANSWER
    {answer}
    
    Evaluate the performance based on the provided context and question.""")
])

# Create the evaluation chain
eval_chain = eval_prompt | judge_llm
# 3. Running the Call
# You can now call this for each of your 20 test cases to generate the data needed for your Day 36 analysis.
def evaluate_run(question, context, answer):
    # This call performs the LLM-as-a-Judge evaluation
    result = eval_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    
    print(f"Relevance: {result.relevance_score}/10")
    print(f"Groundedness: {result.groundedness_score}/10")
    print(f"Reasoning: {result.reasoning}")
    return result
final_response="""
  To run a local LLM like Llama 3.2 3B, you'll need:

  1. Sufficient RAM: At least 8GB of RAM is recommended to prevent memory pressure and ensure smooth performance.
  2. VRAM/RAM: A 3B parameter model in 4-bit quantization typically requires around 2.5GB of VRAM/RAM, leaving room for the OS and background tasks.

  As for CPU inference, since Llama 3.2 3B is a Small Language Model (SLM), it's likely that CPU inference will be relatively efficient compared to larger models. However, I don't have specific information on the exact CPU inference requirements for this model.
  """
user_query= "Local LLM hardware requirements and CPU inference"
retrieved_docs="""
    Llama 3.2 3B is a 'Small Language Model' (SLM). 
        A 3B parameter model in 4-bit quantization typically requires ~2.5GB of VRAM/RAM. 
        This makes it ideal for 8GB RAM laptops, leaving room for the OS and background tasks.
  - |-
    When running local AI on 8GB RAM, memory pressure is the main bottleneck. 
        Closing background apps like Chrome and using lightweight embedding models like 'all-minilm' 
        is essential to prevent the system from using 'Swap memory' on the SSD.# Example usage from your previous "Local LLM" trace:
        """
result = evaluate_run(user_query, retrieved_docs, final_response)
print(result)