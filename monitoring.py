import time
import os
import json
from datetime import datetime

LOG_FILE = 'logs/token_usage.jsonl'

os.makedirs('logs', exist_ok = True)

def log_llm_usage(user_id, stage,response, latency):
    usage = response.usage if hasattr(response, 'usage') else None

    log_entry = {
        'timestamp': str(datetime.utcnow()),
        'user_id': user_id,
        'stage': stage,
        'latency_ms': round(latency * 1000,2),
        'prompt_tokens': getattr(usage,'prompt_tokens', None),
        'completion_tokens': getattr(usage, 'completion_tokens',None),
        'total_tokens': getattr(usage,'total_tokens',None)
    }

    with open(LOG_FILE,'a') as f:
        f.write(json.dumps(log_entry)+ '\n')
        