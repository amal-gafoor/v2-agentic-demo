from config import get_reranker_model

reranker_model = get_reranker_model()

def rerank(query,chunks,top_k=3):

    pairs = [(query,chunk) for chunk in chunks]
    scores = reranker_model.predict(pairs)

    scored_chunk = list(zip(chunks,scores))

    scored_chunk.sort(key=lambda x:x[1], reverse=True)
    top_chunks = [chunk for chunk,score in scored_chunk[:top_k]]

    return top_chunks

