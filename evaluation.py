from retriver import get_documents_for_multiple_queries
from reciprocalFusion import reciprocal_rank_fusion
from rewriter import generate_queries

# The questions and the specific document content that MUST be found
evaluation_set = [
    {
        "question": "What is the specific core count of the Ryzen 5 4600H?",
        "expected_doc": "6-core processor" 
    },
    {
        "question": "How much RAM does a 3B model need in 4-bit?",
        "expected_doc": "~2.5GB of VRAM/RAM"
    },
    {
        "question": "What is the math formula for RRF?",
        "expected_doc": "1/(rank + k)"
    }
]

def evaluate_mrr(eval_set):
    total_reciprocal_rank = 0
    
    for item in eval_set:
        print(f"\nEvaluating Question: {item['question']}")
        
        # 1. Run the Agent Logic
        queries = generate_queries.invoke({'question':(item['question'])})
        all_docs = get_documents_for_multiple_queries(queries)
        ranked_results = reciprocal_rank_fusion(all_docs) # This returns [(content, score), ...]
        
        # 2. Find the rank of the expected document
        rank = None
        for i, (content, score) in enumerate(ranked_results, 1):
            if item['expected_doc'].lower() in content.lower():
                rank = i
                break
        
        # 3. Calculate RR
        if rank:
            rr = 1 / rank
            print(f"✅ Found at Rank: {rank} (RR: {rr:.2f})")
        else:
            rr = 0
            print(f"❌ Document not found in top results (RR: 0)")
            
        total_reciprocal_rank += rr
    
    # 4. Calculate Final MRR
    mrr = total_reciprocal_rank / len(eval_set)
    return mrr
evaluate_mrr(evaluation_set)