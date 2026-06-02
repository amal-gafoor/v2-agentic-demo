# agent/agent.py

import re
from llm_wrapper import agent_llm_call
from agent.tool_registry import TOOL_REGISTRY, call_tool


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
def build_system_prompt() -> str:
    tool_descriptions = ""
    for i, (name, data) in enumerate(TOOL_REGISTRY.items(), 1):
        tool_descriptions += f"{i}. {name} — {data['description']}\n"

    return f"""You are a helpful customer support assistant for a phone case store.
You answer customer questions about products and store policies.

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
- Never guess product prices or policy details — always use a tool first
- If question is about both product AND policy, call both tools one at a time
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
    "no nokia",
    "no samsung",
    "no apple",
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
    """Check if tool returned no useful results."""
    obs_lower = observation.lower()
    return any(signal in obs_lower for signal in NO_RESULT_SIGNALS)


def should_replan(tool_input: str) -> bool:
    """
    Decide if replanning makes sense.
    Only replan if query was specific/complex (3+ words).
    For simple broad queries — product genuinely doesn't exist.
    """
    words = tool_input.strip().split()
    return len(words) >= 3


def build_replan_prompt(tool_name: str, original_input: str, attempt: int) -> str:
    """Tell LLM to try a simpler query."""
    return (
        f"OBSERVATION: The tool '{tool_name}' returned no results "
        f"for query: '{original_input}'.\n\n"
        f"Replan attempt {attempt}/{2}. "
        f"The query was too specific. Try a SIMPLER, BROADER query.\n"
        f"For example: 'Apple iPhone 13 rugged case' → try 'rugged case'\n\n"
        f"Use REPLAN format:\n"
        f"THOUGHT: <why trying simpler query>\n"
        f"REPLAN: {tool_name}\n"
        f"INPUT: <simpler query>"
    )


def build_not_found_prompt(observation: str) -> str:
    """Tell LLM product genuinely doesn't exist — give honest answer."""
    return (
        f"OBSERVATION:\n{observation}\n\n"
        "This product genuinely doesn't exist in our store. "
        "Give a polite FINAL ANSWER telling the customer we don't carry this, "
        "and suggest they ask about similar products we might have."
    )


# ─────────────────────────────────────────────
# REACT LOOP WITH REPLANNING
# ─────────────────────────────────────────────
def run_react_agent(
    user_query: str,
    user_id: str = "agent",
    history: list = None,
    max_iterations: int = 8,
    max_replans: int = 2
) -> str:

    history = history or []

    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-6:]
    ]

    turns = (
        [{"role": "system", "content": build_system_prompt()}]
        + history_messages
        + [{"role": "user", "content": user_query}]
    )

    # Track replans per tool
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

        has_action       = bool(re.search(r"ACTION:\s*\w+",  llm_output))
        has_replan       = bool(re.search(r"REPLAN:\s*\w+",  llm_output))
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
            answer = llm_output.split("FINAL ANSWER:")[-1].strip()
            print(f"[Agent] Final answer ready")
            return answer

        # ── REPLAN — retry with different query ──
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

            # Max replans reached — force honest answer
            if replan_counts[tool_name] > max_replans:
                print(f"[Agent] Max replans reached for {tool_name}")
                turns.append({"role": "assistant", "content": llm_output})
                turns.append({
                    "role": "user",
                    "content": (
                        f"You have tried {max_replans} times and found nothing. "
                        "Give an honest FINAL ANSWER — we don't carry this product. "
                        "Suggest the customer ask about other products."
                    )
                })
                continue

            print(f"[Agent] REPLAN {replan_counts[tool_name]}/{max_replans} | "
                  f"Tool: {tool_name} | New input: {tool_input}")

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

        # ── Decide: replan or continue? ──
        if is_empty_result(observation):
            if should_replan(tool_input):
                print(f"[Agent] Specific query failed — triggering replan")
                turns.append({
                    "role": "user",
                    "content": build_replan_prompt(tool_name, tool_input, 1)
                })
            else:
                print(f"[Agent] Simple query, nothing found — product doesn't exist")
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