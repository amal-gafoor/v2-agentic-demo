# agent/agent.py

import re
from llm_wrapper import agent_llm_call
from agent.tool_registry import TOOL_REGISTRY, call_tool
from profile_store import (
    needs_purchase_context,
    get_purchase_context
)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
def build_system_prompt(purchase_context: str = "") -> str:
    tool_descriptions = ""
    for i, (name, data) in enumerate(TOOL_REGISTRY.items(), 1):
        tool_descriptions += f"{i}. {name} — {data['description']}\n"

    # Only inject purchase context when relevant
    purchase_section = ""
    if purchase_context:
        purchase_section = f"\n{purchase_context}\n"

    return f"""You are a helpful customer support assistant.
You help customers with product queries and store policies.
{purchase_section}
You have access to these tools:
{tool_descriptions}
STRICT FORMAT — follow exactly every single time:

When you need information from a tool:
THOUGHT: <your reasoning>
ACTION: <tool_name>
INPUT: <your search query>

When a tool returns no results and the query was specific — try simpler:
THOUGHT: <why retrying with simpler query>
REPLAN: <tool_name>
INPUT: <simpler broader query>

When you have enough information to answer:
THOUGHT: I have enough information to answer.
FINAL ANSWER: <your friendly, clear answer>

Rules:
- Always write THOUGHT before ACTION, REPLAN, or FINAL ANSWER
- Never guess product details or policy — always use a tool first
- If question covers both product AND policy, call both tools one at a time
- Base your FINAL ANSWER only on what tools returned
- ONE RESPONSE = ONE ACTION or ONE REPLAN or ONE FINAL ANSWER — never mix
- You MUST wait for OBSERVATION before writing FINAL ANSWER
- Maximum 2 replans per tool — after that give honest FINAL ANSWER
- Keep FINAL ANSWER short, friendly, and helpful
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
NO_RESULT_SIGNALS = [
    "no_relevant_results",
    "no relevant products",
    "no relevant",
    "not available",
    "couldn't find",
    "could not find",
    "no information",
    "no cases available",
    "no stock",
    "nothing found",
    "not found",
    "no results",
]

def is_empty_result(observation: str) -> bool:
    obs_lower = observation.lower()
    return any(signal in obs_lower for signal in NO_RESULT_SIGNALS)


def should_replan(tool_input: str) -> bool:
    # Only replan if query was specific (3+ words)
    return len(tool_input.strip().split()) >= 3


def build_replan_prompt(tool_name: str, original_input: str, attempt: int) -> str:
    return (
        f"OBSERVATION: The tool '{tool_name}' returned no results "
        f"for query: '{original_input}'.\n\n"
        f"Replan attempt {attempt}/2. "
        f"Try a SIMPLER, BROADER query.\n"
        f"Example: 'Apple iPhone 13 rugged case' → try 'rugged case'\n\n"
        f"THOUGHT: <why trying simpler query>\n"
        f"REPLAN: {tool_name}\n"
        f"INPUT: <simpler query>"
    )


def build_not_found_prompt(observation: str) -> str:
    return (
        f"OBSERVATION:\n{observation}\n\n"
        "This product doesn't exist in our store. "
        "Give a polite FINAL ANSWER saying we don't carry this, "
        "and invite the customer to ask about other products."
    )


# ─────────────────────────────────────────────
# REACT LOOP
# ─────────────────────────────────────────────
def run_react_agent(
    user_query: str,
    user_id: str = "agent",
    history: list = None,
    profile: dict = None,
    max_iterations: int = 8,
    max_replans: int = 2
) -> str:

    history = history or []
    profile = profile or {}

    # ── Decide whether to inject purchase context ──
    # Only inject if customer is asking about past purchases
    purchase_context = ""
    if profile and needs_purchase_context(user_query):
        purchase_context = get_purchase_context(profile)
        if purchase_context:
            print(f"[Agent] Purchase context injected for query: {user_query[:50]}")
        else:
            print(f"[Agent] No purchase history found for context")
    else:
        print(f"[Agent] No purchase context needed for this query")

    # Last 6 messages for current convo context
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-6:]
    ]

    # Build turns — system prompt (with/without purchase context) + history + query
    turns = (
        [{"role": "system", "content": build_system_prompt(purchase_context)}]
        + history_messages
        + [{"role": "user", "content": user_query}]
    )

    replan_counts: dict = {}

    for iteration in range(max_iterations):
        print(f"\n[Agent] Iteration {iteration + 1}")

        llm_output = agent_llm_call(
            user_id=user_id,
            messages=turns,
            temperature=0.1,
            max_tokens=400
        )

        if llm_output is None:
            return "Sorry, I had trouble processing that. Please try again."

        print(f"[Agent] LLM:\n{llm_output}\n")

        has_action       = bool(re.search(r"ACTION:\s*\w+", llm_output))
        has_replan       = bool(re.search(r"REPLAN:\s*\w+", llm_output))
        has_final_answer = "FINAL ANSWER:" in llm_output

        # ── Reject mixed response ──
        if (has_action or has_replan) and has_final_answer:
            print("[Agent] WARNING: mixed response — rejecting")
            turns.append({"role": "assistant", "content": llm_output})
            turns.append({
                "role": "user",
                "content": (
                    "ONE response = ONE action only.\n"
                    "Do not write FINAL ANSWER in the same response as ACTION or REPLAN.\n"
                    "Call the tool first. Wait for OBSERVATION. Then give FINAL ANSWER."
                )
            })
            continue

        # ── FINAL ANSWER ──
        if has_final_answer:
            return llm_output.split("FINAL ANSWER:")[-1].strip()

        # ── REPLAN ──
        replan_match = re.search(r"REPLAN:\s*(\w+)", llm_output)
        if has_replan and replan_match:
            input_match = re.search(r"INPUT:\s*(.+)", llm_output)
            if not input_match:
                turns.append({"role": "assistant", "content": llm_output})
                turns.append({
                    "role": "user",
                    "content": "REPLAN must include INPUT. Provide a new simpler search query."
                })
                continue

            tool_name  = replan_match.group(1).strip()
            tool_input = input_match.group(1).strip()
            replan_counts[tool_name] = replan_counts.get(tool_name, 0) + 1

            if replan_counts[tool_name] > max_replans:
                turns.append({"role": "assistant", "content": llm_output})
                turns.append({
                    "role": "user",
                    "content": (
                        f"You have tried {max_replans} times and found nothing. "
                        "Give an honest FINAL ANSWER."
                    )
                })
                continue

            print(f"[Agent] REPLAN {replan_counts[tool_name]}/{max_replans} | "
                  f"Tool: {tool_name} | Input: {tool_input}")

            observation = call_tool(tool_name, tool_input, user_id)
            print(f"[Agent] Replan observation: {observation[:200]}...")
            turns.append({"role": "assistant", "content": llm_output})

            if is_empty_result(observation):
                if replan_counts[tool_name] < max_replans:
                    turns.append({
                        "role": "user",
                        "content": build_replan_prompt(
                            tool_name, tool_input,
                            replan_counts[tool_name] + 1
                        )
                    })
                else:
                    turns.append({
                        "role": "user",
                        "content": build_not_found_prompt(observation)
                    })
            else:
                turns.append({
                    "role": "user",
                    "content": (
                        f"OBSERVATION:\n{observation}\n\n"
                        "Results found. Now give your FINAL ANSWER."
                    )
                })
            continue

        # ── Regular ACTION ──
        action_match = re.search(r"ACTION:\s*(\w+)", llm_output)
        input_match  = re.search(r"INPUT:\s*(.+)",   llm_output)

        if not action_match or not input_match:
            turns.append({"role": "assistant", "content": llm_output})
            turns.append({
                "role": "user",
                "content": (
                    "Follow the exact format:\n"
                    "THOUGHT: ...\nACTION: tool_name\nINPUT: query\n\n"
                    "Or:\nTHOUGHT: ...\nFINAL ANSWER: answer"
                )
            })
            continue

        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        print(f"[Agent] Tool: {tool_name} | Input: {tool_input}")

        observation = call_tool(tool_name, tool_input, user_id)
        print(f"[Agent] Observation: {observation[:200]}...")
        turns.append({"role": "assistant", "content": llm_output})

        if is_empty_result(observation):
            if should_replan(tool_input):
                turns.append({
                    "role": "user",
                    "content": build_replan_prompt(tool_name, tool_input, 1)
                })
            else:
                turns.append({
                    "role": "user",
                    "content": build_not_found_prompt(observation)
                })
        else:
            turns.append({
                "role": "user",
                "content": (
                    f"OBSERVATION:\n{observation}\n\n"
                    "Now give your FINAL ANSWER based on this. "
                    "Do not call more tools unless absolutely necessary."
                )
            })

    return "I wasn't able to find a complete answer. Could you rephrase your question?"