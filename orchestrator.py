# orchestrator.py
# The brain — classifies tasks, generates plans, routes to executor

import json
import re
import time
from llm_wrapper import agent_llm_call, safe_llm_call
from memory_store import (
    set_plan, get_plan, save_step_result,
    mark_plan_done, clear_working_memory,
    add_goal, get_active_goals
)


# ─────────────────────────────────────────────
# TASK TYPES
# ─────────────────────────────────────────────
SIMPLE      = "simple"       # answer directly, no tools
SINGLE_TOOL = "single_tool"  # one tool call, existing ReAct loop
MULTI_STEP  = "multi_step"   # multiple steps, needs plan
PROACTIVE   = "proactive"    # store for future execution


# ─────────────────────────────────────────────
# CLASSIFICATION PROMPT
# ─────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are an intelligent task classifier for a personal/company assistant.

Given a user request, classify it into exactly one of these task types:

SIMPLE      — Can be answered directly without any tools or external data.
              Examples: "what is 2+2", "what's today's date", "what does RAG mean"

SINGLE_TOOL — Needs exactly one tool call to answer.
              Examples: "do you have leather cases", "what is your return policy",
                        "what's the price of the rugged case"

MULTI_STEP  — Needs multiple steps, tools, or actions to complete.
              Examples: "find a rugged case under $30 and check return policy",
                        "book me a flight to Dubai next Friday cheapest option",
                        "compare all iPhone cases and tell me the best one",
                        "find me a case, check if it's in stock, and what's shipping time"

PROACTIVE   — A future task, reminder, or scheduled action. Not immediate.
              Examples: "remind me to follow up with John tomorrow",
                        "alert me when stock drops below 10",
                        "check flight prices every day this week"

Respond in this EXACT format — nothing else:
TASK_TYPE: <simple|single_tool|multi_step|proactive>
REASONING: <one sentence why>
DEADLINE: <date or time if proactive, otherwise "none">
"""


def classify_task(query: str, user_id: str, history: list) -> dict:
    """
    LLM classifies the task type.
    Returns dict with task_type, reasoning, deadline.
    """
    # Build messages — include last 4 history messages for context
    history_text = ""
    if history:
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"{CLASSIFICATION_PROMPT}\n\nConversation context:\n{history_text}\nUser request: {query}"

    messages = [
        {"role": "system", "content": "You are a task classifier. Follow the exact format."},
        {"role": "user",   "content": prompt}
    ]

    response = agent_llm_call(
        user_id=user_id,
        messages=messages,
        temperature=0.0,   # deterministic — classification should be consistent
        max_tokens=150
    )

    if not response:
        return {"task_type": SINGLE_TOOL, "reasoning": "fallback", "deadline": None}

    # Parse response
    task_type_match = re.search(r"TASK_TYPE:\s*(\w+)", response)
    reasoning_match = re.search(r"REASONING:\s*(.+)",  response)
    deadline_match  = re.search(r"DEADLINE:\s*(.+)",   response)

    task_type = task_type_match.group(1).lower().strip() if task_type_match else SINGLE_TOOL
    reasoning = reasoning_match.group(1).strip()         if reasoning_match else ""
    deadline  = deadline_match.group(1).strip()          if deadline_match  else None

    # Validate task type
    valid_types = [SIMPLE, SINGLE_TOOL, MULTI_STEP, PROACTIVE]
    if task_type not in valid_types:
        task_type = SINGLE_TOOL

    # Clean deadline
    if deadline and deadline.lower() in ["none", "n/a", ""]:
        deadline = None

    print(f"[Orchestrator] Task: {task_type} | Reason: {reasoning[:60]}")
    return {
        "task_type": task_type,
        "reasoning": reasoning,
        "deadline":  deadline
    }


# ─────────────────────────────────────────────
# PLAN GENERATION
# ─────────────────────────────────────────────
PLANNING_PROMPT = """You are a planning assistant. Break down a complex task into clear steps.

Rules:
- Each step must be a single, specific action
- Steps should be in logical order
- Maximum 5 steps — keep it focused
- Each step should map to one tool call or one clear action
- Write steps as instructions, not questions

Available tools:
- search_products: search for products by name, features, price
- search_policies: search for store policies (return, shipping, payment)

Respond ONLY with a JSON array of steps. Nothing else. No explanation.
Example: ["Search for rugged iPhone cases", "Check return policy for the found products"]
"""

def generate_plan(query: str, user_id: str, history: list) -> list:
    """
    LLM generates a step-by-step plan for a multi-step task.
    Returns list of step strings.
    """
    history_text = ""
    if history:
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"{PLANNING_PROMPT}\n\nContext:\n{history_text}\nTask: {query}\n\nSteps (JSON array only):"

    response = safe_llm_call(
        user_id=user_id,
        stage="plan_generation",
        prompt=prompt,
        temperature=0.1,
        max_tokens=200
    )

    if not response:
        # Fallback — treat as single tool
        return [query]

    try:
        # Clean response and parse JSON
        clean = response.strip()
        # Find JSON array in response
        match = re.search(r'\[.*?\]', clean, re.DOTALL)
        if match:
            steps = json.loads(match.group())
            print(f"[Orchestrator] Plan generated: {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                print(f"  Step {i}: {step}")
            return steps
    except Exception as e:
        print(f"[Plan Parse Error] {e}")

    # Fallback
    return [query]


# ─────────────────────────────────────────────
# PLAN EXECUTION
# ─────────────────────────────────────────────
def execute_plan(
    steps: list,
    original_query: str,
    user_id: str,
    session: dict,
    history: list,
    profile: dict
) -> str:
    """
    Executes each step of the plan using the ReAct agent.
    Synthesizes all results into a final answer.
    """
    from agent.agent import run_react_agent

    all_results = []

    for i, step in enumerate(steps):
        print(f"\n[Orchestrator] Executing step {i+1}/{len(steps)}: {step}")

        # Run agent for this step
        # Pass previous results as context so agent knows what was found
        step_context = ""
        if all_results:
            step_context = "Previous steps completed:\n"
            for prev in all_results:
                step_context += f"- {prev['step']}: {prev['result'][:200]}\n"
            step_context += f"\nNow execute: {step}"
        else:
            step_context = step

        result = run_react_agent(
            user_query=step_context,
            user_id=user_id,
            history=history,
            profile=profile
        )

        all_results.append({"step": step, "result": result})
        save_step_result(session, step, result)

        print(f"[Orchestrator] Step {i+1} result: {result[:100]}...")

    # Synthesize all results into one final answer
    final = synthesize_results(original_query, all_results, user_id)
    mark_plan_done(session)
    clear_working_memory(session)

    return final


def synthesize_results(original_query: str, results: list, user_id: str) -> str:
    """
    Combines results from all plan steps into one coherent answer.
    """
    results_text = ""
    for i, r in enumerate(results, 1):
        results_text += f"Step {i} — {r['step']}:\n{r['result']}\n\n"

    prompt = f"""A user asked: "{original_query}"

The following information was gathered step by step:

{results_text}

Now write a single, clear, friendly response that answers the user's original question
using all the information gathered above.
Do not repeat step labels. Just give a natural, complete answer.
"""

    response = safe_llm_call(
        user_id=user_id,
        stage="synthesis",
        prompt=prompt,
        temperature=0.2,
        max_tokens=400
    )

    return response or results[-1]['result']


# ─────────────────────────────────────────────
# PROACTIVE TASK HANDLER
# ─────────────────────────────────────────────
def handle_proactive(query: str, deadline: str, session: dict) -> str:
    """
    Stores a proactive task in goal memory.
    Returns confirmation message to user.
    """
    goal_id = add_goal(session, query, deadline)

    deadline_text = f" by {deadline}" if deadline else ""
    return (
        f"Got it! I've noted that down{deadline_text}. "
        f"I'll take care of it when the time comes. "
        f"(Goal ID: {goal_id})"
    )


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR — entry point
# ─────────────────────────────────────────────
def run_orchestrator(
    user_query: str,
    user_id: str,
    session: dict,
    history: list,
    profile: dict
) -> str:
    """
    Main entry point. Called by app.py instead of agent directly.

    Flow:
    1. Classify the task
    2. Route based on task type:
       - simple      → answer directly
       - single_tool → run ReAct agent as before
       - multi_step  → generate plan → execute step by step
       - proactive   → store in goal memory
    """
    from agent.agent import run_react_agent

    print(f"\n[Orchestrator] Query: {user_query[:80]}")

    # Step 1 — classify
    classification = classify_task(user_query, user_id, history)
    task_type      = classification['task_type']
    deadline       = classification['deadline']

    # Step 2 — route

    # ── SIMPLE — answer directly ──
    if task_type == SIMPLE:
        print("[Orchestrator] Route: simple → direct answer")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user",   "content": user_query}
        ]
        response = agent_llm_call(
            user_id=user_id,
            messages=messages,
            temperature=0.2,
            max_tokens=200
        )
        return response or "I'm not sure about that."

    # ── SINGLE TOOL — existing ReAct loop ──
    elif task_type == SINGLE_TOOL:
        print("[Orchestrator] Route: single_tool → ReAct agent")
        return run_react_agent(
            user_query=user_query,
            user_id=user_id,
            history=history,
            profile=profile
        )

    # ── MULTI STEP — plan + execute ──
    elif task_type == MULTI_STEP:
        print("[Orchestrator] Route: multi_step → plan + execute")

        # Generate plan
        steps = generate_plan(user_query, user_id, history)

        if len(steps) <= 1:
            # Plan collapsed to one step — treat as single tool
            print("[Orchestrator] Plan has 1 step — routing to ReAct agent")
            return run_react_agent(
                user_query=user_query,
                user_id=user_id,
                history=history,
                profile=profile
            )

        # Save plan to working memory
        set_plan(session, user_query, steps)

        # Execute plan
        return execute_plan(
            steps=steps,
            original_query=user_query,
            user_id=user_id,
            session=session,
            history=history,
            profile=profile
        )

    # ── PROACTIVE — store for later ──
    elif task_type == PROACTIVE:
        print("[Orchestrator] Route: proactive → goal memory")
        return handle_proactive(user_query, deadline, session)

    # ── FALLBACK ──
    else:
        print("[Orchestrator] Route: fallback → ReAct agent")
        return run_react_agent(
            user_query=user_query,
            user_id=user_id,
            history=history,
            profile=profile
        )