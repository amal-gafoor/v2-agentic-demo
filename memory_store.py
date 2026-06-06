# memory_store.py

import os
import json
import time
from config import SESSION_DIR, MAX_HISTORY

os.makedirs(SESSION_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DEFAULT STRUCTURE
# ─────────────────────────────────────────────
def default_session() -> dict:
    return {
        "conversation": {
            "messages": []
        },
        "session": {
            "order_state": "idle",
            "order_data": {
                "product":  None,
                "quantity": None,
                "template": None
            },
            "last_active": None
        },
        "scratchpad": {
            "current_run": []
        },

        # ── NEW: working memory ──
        # tracks multi-step plan execution within a session
        "working_memory": {
            "current_goal":  None,   # what the user wants to achieve
            "plan":          [],     # list of step strings
            "current_step":  0,      # index of step being executed
            "status":        "idle", # idle | planning | executing | done | failed
            "results":       []      # result from each step
        },

        # ── NEW: goal memory ──
        # stores proactive tasks for future execution
        "goal_memory": {
            "active_goals": []
            # each goal:
            # {
            #   "id":       "goal_001",
            #   "goal":     "remind user to follow up with John",
            #   "deadline": "2026-06-10",
            #   "status":   "pending",  # pending | in_progress | done | cancelled
            #   "created":  "2026-06-01"
            # }
        }
    }


# ─────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────
def get_path(user_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{user_id}.json")


def load_memory(user_id: str) -> dict:
    path = get_path(user_id)

    if not os.path.exists(path):
        return default_session()

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Safety — fill any missing keys with defaults
        defaults = default_session()
        for key in defaults:
            if key not in data:
                data[key] = defaults[key]

        return data

    except Exception as e:
        print(f"[MEMORY LOAD ERROR] {e}")
        return default_session()


def save_memory(user_id: str, session: dict) -> None:
    path = get_path(user_id)

    try:
        # Sliding window — keep only last MAX_HISTORY messages
        messages = session['conversation']['messages']
        if len(messages) > MAX_HISTORY:
            session['conversation']['messages'] = messages[-MAX_HISTORY:]

        session['session']['last_active'] = time.time()

        with open(path, 'w') as f:
            json.dump(session, f, indent=2)

    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")


# ─────────────────────────────────────────────
# CONVERSATION HELPERS
# ─────────────────────────────────────────────
def add_message(session: dict, role: str, content: str) -> None:
    session['conversation']['messages'].append({
        "role":    role,
        "content": content,
        "time":    time.time()
    })


def get_history(session: dict) -> list:
    return session['conversation']['messages']


def clear_scratchpad(session: dict) -> None:
    session['scratchpad']['current_run'] = []


# ─────────────────────────────────────────────
# WORKING MEMORY HELPERS
# used by orchestrator to track plan execution
# ─────────────────────────────────────────────
def set_plan(session: dict, goal: str, steps: list) -> None:
    """
    Save a new plan to working memory.
    Called by orchestrator when a multi-step task is detected.
    """
    session['working_memory'] = {
        "current_goal": goal,
        "plan":         steps,
        "current_step": 0,
        "status":       "executing",
        "results":      []
    }
    print(f"[Working Memory] Plan set: {len(steps)} steps for goal: {goal[:50]}")


def get_plan(session: dict) -> dict:
    """Returns the full working memory."""
    return session['working_memory']


def save_step_result(session: dict, step: str, result: str) -> None:
    """Save the result of a completed step."""
    session['working_memory']['results'].append({
        "step":   step,
        "result": result
    })
    session['working_memory']['current_step'] += 1
    print(f"[Working Memory] Step {session['working_memory']['current_step']} saved")


def mark_plan_done(session: dict) -> None:
    """Mark plan as completed and clear working memory."""
    session['working_memory']['status'] = "done"
    print(f"[Working Memory] Plan completed")


def clear_working_memory(session: dict) -> None:
    """Reset working memory after plan is done."""
    session['working_memory'] = {
        "current_goal": None,
        "plan":         [],
        "current_step": 0,
        "status":       "idle",
        "results":      []
    }


def is_plan_active(session: dict) -> bool:
    """Check if a plan is currently being executed."""
    wm = session['working_memory']
    return wm['status'] == "executing" and len(wm['plan']) > 0


# ─────────────────────────────────────────────
# GOAL MEMORY HELPERS
# used by orchestrator to store proactive tasks
# ─────────────────────────────────────────────
def add_goal(session: dict, goal: str, deadline: str = None) -> str:
    """
    Store a proactive goal for future execution.
    Returns the goal ID.
    """
    goal_id = f"goal_{int(time.time())}"
    session['goal_memory']['active_goals'].append({
        "id":       goal_id,
        "goal":     goal,
        "deadline": deadline,
        "status":   "pending",
        "created":  time.strftime("%Y-%m-%d")
    })
    print(f"[Goal Memory] Goal saved: {goal[:50]}")
    return goal_id


def get_active_goals(session: dict) -> list:
    """Returns all pending goals."""
    return [
        g for g in session['goal_memory']['active_goals']
        if g['status'] == 'pending'
    ]


def mark_goal_done(session: dict, goal_id: str) -> None:
    """Mark a specific goal as completed."""
    for goal in session['goal_memory']['active_goals']:
        if goal['id'] == goal_id:
            goal['status'] = 'done'
            break