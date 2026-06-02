# prompts.py — update only COMPRESS_PROMPT
 
COMPRESS_PROMPT = """
Extract ONLY the information relevant to the customer question.
Return concise bullet points.
Do not include unrelated sections.
 
IMPORTANT: If the retrieved knowledge contains NO relevant information
for the customer question — respond with exactly this single word:
NO_RELEVANT_RESULTS
 
Customer Question:
{question}
 
Retrieved Knowledge:
{context}
"""
