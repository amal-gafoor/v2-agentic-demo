from .embeddings import get_embeddings
import numpy as np
import faiss


def retriever(query,index,documents,domain =None,top_k=10):
    query_vector = np.array([get_embeddings(query)]).astype("float32")
    faiss.normalize_L2(query_vector)

    search_k = top_k*3

    scores,indices = index.search(query_vector,search_k) 
    results = []    
    
    for i in indices[0]:
        if i == -1:
            continue
        chunk,meta=documents[i]

        if domain is None or meta['domain'] == domain:
            results.append(chunk)

        if len(results) >= top_k:
            break
    
    return results 