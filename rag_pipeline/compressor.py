from prompts import COMPRESS_PROMPT
from llm_wrapper import safe_llm_call


def compress(question,retrieved_chunks,user_id):
    context = '\n\n'.join(retrieved_chunks)
    prompt=COMPRESS_PROMPT.format(
        question=question,
        context=context
    )

    response = safe_llm_call(
        user_id = user_id,
        stage = 'compress',
        prompt = prompt,
        temperature = 0.2,
        max_tokens = 150
    )
    if response is None:
        return '\n'.join(retrieved_chunks[:3])
    return response
    

