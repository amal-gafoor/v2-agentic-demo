
from prompts import GENERATION_PROMPT
import time
from llm_wrapper import safe_llm_call

def generator(question,compressed_context,history,user_id):
    history_text = ''
    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"
 
    prompt = GENERATION_PROMPT.format(
        history=history_text,
        question=question,
        context=compressed_context
    )

    response = safe_llm_call(
        user_id = user_id,
        stage = 'compress',
        prompt = prompt,
        temperature = 0.7,
        max_tokens = 100
    )
    if response is None:
        return "I'm checking that for you. Could you clarify slightly?"
    return response