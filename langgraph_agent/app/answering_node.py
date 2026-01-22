
from agent_state import AgentState
from model import llm
from retry_func import node_with_retry
from schemas import FinalResponse
from decorators import validate_llm_output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define the generation logic as a separate, decorated function
@validate_llm_output(FinalResponse)
def generate_structured_answer(context, question):
    # Update prompt to strictly request JSON matching FinalResponse schema
    template = """You are a helpful assistant. Answer the question based ONLY on the context.
    
    You must return a JSON object with these exact keys:
    - "answer": The answer text.
    - "citations": A list of direct quotes or document names used.
    - "confidence_score": A float between 0.0 and 1.0.

    Context: {context}
    Question: {question}
    
    JSON Output:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return response


def answering_node(state: AgentState):
    def process(s):
        if not s.get("reranked_docs"):
            return {"final_answer": "I'm sorry, I couldn't find relevant info."}
        
        question = s["question"]
        # Join docs with a delimiter for better context separation
        context = "\n---\n".join([doc for doc in s["reranked_docs"]])
        
        # 2. Call the decorated function (Auto-validates JSON)
        try:
            response_obj = generate_structured_answer(context, question)
            
            # 3. Return the validated fields
            return {
                "final_answer": response_obj.answer,
                "citations": response_obj.citations, # New state field needed?
                "confidence": response_obj.confidence_score
            }
        except Exception as e:
            # Retry logic from 'node_with_retry' will catch this re-raised error
            raise e 

    return node_with_retry(process, state, "Answerer")
