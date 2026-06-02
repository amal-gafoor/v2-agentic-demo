import time 
from config  import get_groq_client,GROQ_MODEL
from monitoring import log_llm_usage

client = get_groq_client()

def safe_llm_call(user_id, stage, prompt, temperature, max_tokens):

    try:
        start = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature= temperature,
            max_tokens=max_tokens
        )

        latency = time.time() - start

        log_llm_usage(user_id, stage, response, latency)

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f'[LLM ERROR] stage:{stage} | User: {user_id} | Error: {e}')
        return None
    

# ─────────────────────────────────────────────
# NEW — used by ReAct agent only
# accepts full messages list with roles
# ─────────────────────────────────────────────
def agent_llm_call(user_id, messages, temperature=0.1, max_tokens=500):
    """
    For the ReAct agent loop.
    Accepts a proper messages list:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "OBSERVATION: ..."},
    ]
    """
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        latency = time.time() - start
        log_llm_usage(user_id, 'react_agent', response, latency)
        return response.choices[0].message.content.strip()
 
    except Exception as e:
        print(f'[AGENT LLM ERROR] | User: {user_id} | Error: {e}')
        return None