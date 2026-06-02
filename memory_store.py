import os
import json
import time
from config import SESSION_DIR, MAX_HISTORY

os.makedirs(SESSION_DIR,exist_ok=True)

def get_path(user_id):
    return os.path.join(SESSION_DIR, f"{user_id}.json")

def default_session():
    return {
        "conversation": {
            "messages": []
        },
        "session": {
            "order_state": "idle",        # # idle | collecting | confirming | placed
            "order_data": {
                "product": None,
                "quantity": None,
                "template": None
            },
            "last_active": None
        },
        "scratchpad": {
            "current_run": []             # cleared after every agent turn 
        }
    }

def load_memory(user_id):
    path = get_path(user_id)

    if not os.path.exists(path):
        return default_session()
                
    try:
        with open(path,'r') as f:
            data= json.load(f)

        # Safety checks — if any key missing, fill with defaults
        defaults = default_session()
        for key in defaults:
            if key not in data:
                data[key] = defaults[key]

        return data
        
            
    except Exception as e:
        print(f"[MEMORY LOAD ERROR] {e}")
        return default_session

def save_memory(user_id,session):
    path = get_path(user_id)

    try:
        # Sliding window — keep only last MAX_HISTORY messages
        messages = session['conversation']['messages']
        if len(messages) > MAX_HISTORY:
            session['conversation']['messages'] = messages[-MAX_HISTORY:]

        # Update last active timestamp
        session['session']['last_active'] = time.time()

        with open(path,'w') as f:
            json.dump(session,f,indent=2)

    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")


# ─────────────────────────────────────────────
# CONVERSATION HELPERS
# ─────────────────────────────────────────────
def add_message(session: dict, role: str, content: str) -> None:
    """Add a message to conversation history."""
    session['conversation']['messages'].append({
        "role":    role,
        "content": content,
        "time":    time.time()
    })

def get_history(session: dict) -> list:
    """Get conversation messages for passing to LLM."""
    return session['conversation']['messages']

# ─────────────────────────────────────────────
# SCRATCHPAD HELPERS
# ─────────────────────────────────────────────
def add_to_scratchpad(session: dict, type: str, content: dict) -> None:
    """
    Track what agent did in current run.
    type: 'thought' | 'action' | 'observation'
    """
    session['scratchpad']['current_run'].append({
        "type":    type,
        "content": content,
        "time":    time.time()
    })

def clear_scratchpad(session: dict) -> None:
    """Clear scratchpad after every agent turn."""
    session['scratchpad']['current_run'] = []


# ─────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────
def get_order_state(session: dict) -> str:
    return session['session']['order_state']

def set_order_state(session: dict, state: str) -> None:
    """state: idle | collecting | confirming | placed"""
    session['session']['order_state'] = state

def update_order_data(session: dict, key: str, value) -> None:
    """Update a specific field in order_data."""
    session['session']['order_data'][key] = value

def reset_order(session: dict) -> None:
    """Reset order back to idle — after placed or cancelled."""
    session['session']['order_state'] = 'idle'
    session['session']['order_data'] = {
        "product":  None,
        "quantity": None,
        "template": None
    }
