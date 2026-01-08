def reciprocal_rank_fusion(results_list: list[list], k=60):
    """
    results_list: A list containing 3 lists of retrieved Documents
    k: A constant (default 60) that smooths the ranking
    """
    fused_scores = {}
    
    # 1. Loop through each of the 3 result sets
    for docs in results_list:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            # 2. If it's a new document, start score at 0
            if content not in fused_scores:
                fused_scores[content] = 0
            
            # 3. The RRF Formula: 1 / (rank + k)
            # rank + 1 because rank starts at 0
            fused_scores[content] += 1 / (rank + 1 + k)
            
    # 4. Sort by score in descending order
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked_results
