# agent/tools/search_products.py

from rag_pipeline.vector_store import load_vector_store, build_vector_store
from rag_pipeline.retriever import retriever
from rag_pipeline.reranker import rerank
from rag_pipeline.compressor import compress


def search_products(query: str, user_id: str = "agent", history: list = None) -> str:
    """
    Tool: search_products

    Runs RAG pipeline for a product query:
    Retriever → Reranker → Compressor

    Returns compressed context to the ReAct agent.
    The agent generates the final answer itself.

    Use when customer asks about:
    - Product details, features, specs
    - Price of a product
    - Which product is best for their need
    - Comparison between products
    """

    # Step 1 — load vector store (cached)
    index, documents = load_vector_store()
    if index is None:
        index, documents = build_vector_store()

    # Step 2 — retrieve top chunks (products only)
    retrieved_chunks = retriever(
        query,
        index,
        documents,
        domain='product'
    )

    if not retrieved_chunks:
        return "No relevant products found for this query."

    # Step 3 — rerank: top 9 → top 3
    try:
        reranked_chunks = rerank(
            query,
            retrieved_chunks,
            top_k=3
        )
    except Exception as e:
        print(f"[RERANK ERROR] {e}")
        reranked_chunks = retrieved_chunks[:3]

    # Step 4 — compress and return context to agent
    compressed_context = compress(
        query,
        reranked_chunks,
        user_id
    )

    return compressed_context