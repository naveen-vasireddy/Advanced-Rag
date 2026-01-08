
from reciprocalFusion import reciprocal_rank_fusion
from retriver import get_documents_for_multiple_queries
from rewriter import generate_queries
from agent import generate_final_answer
question = "Local LLM hardware requirements and CPU inference"
final_chain = generate_queries | get_documents_for_multiple_queries

results = final_chain.invoke({"question": question})
print(f"\n📄 Retrieved Documents for all queries: {results}")
# 3. Merge and Rank
top_docs = reciprocal_rank_fusion(results)

# 4. Print the "winner"
print(f"Top Documents by RRF: {top_docs}")

# 5. Generate Final Answer
final_answer = generate_final_answer(question, top_docs)   
print(f"\n🤖 Final Answer:\n{final_answer}")
