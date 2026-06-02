

COMPRESS_PROMPT = """
Extract ONLY the information relevant to the customer question.
Return concise bullet points.
Do not include unrelated sections.

Customer Question:
{question}

Retrieved Knowledge:
{context}
"""
