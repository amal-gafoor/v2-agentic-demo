# agent/agent.py

import re
from llm_wrapper import agent_llm_call
from agent.tool_registry import TOOL_REGISTRY, call_tool


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
 
When a tool returns no results and you want to try again:
THOUGHT: <why you are retrying and what different query you will try>
REPLAN: <tool_name>
INPUT: <your new different search query>
 
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
- If tool returns nothing — use REPLAN with a different, simpler query
- Maximum 2 replans per tool — after that give honest FINAL ANSWER
- Keep FINAL ANSWER short, friendly, and helpful
"""
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
NO_RESULT_SIGNALS = [
    "no relevant products found",
    "no results",
    "nothing found",
    "not found",
    "no stock",
    "couldn't find",
    "could not find",
    "no information",
    "no data",
]

def is_empty_result(observation: str) -> bool:
    """Check if tool returned no useful results."""
    obs_lower = observation.lower()
    return any(signal in obs_lower for signal in NO_RESULT_SIGNALS)

def build_replan_prompt(tool_name: str, original_input: str, attempt: int) -> str:
    """Tell LLM its tool returned nothing and ask it to replan."""
    return (
        f"OBSERVATION: The tool '{tool_name}' returned no results "
        f"for query: '{original_input}'.\n\n"
        f"This is replan attempt {attempt}. "
        f"Think of a DIFFERENT, simpler search query and try again using REPLAN.\n"
        f"Or if you've already tried multiple times, give a honest FINAL ANSWER "
        f"saying the product/info wasn't found.\n\n"
        f"Remember: REPLAN format is:\n"
        f"THOUGHT: <why trying different query>\n"
        f"REPLAN: <tool_name>\n"
        f"INPUT: <new simpler query>"
    )

# ─────────────────────────────────────────────
# REACT LOOP WITH REPLANNING
# ─────────────────────────────────────────────
def run_react_agent(
    user_query: str,
    user_id: str = "agent",
    history: list = None,
    max_iterations: int = 8,    # slightly higher to allow replans
    max_replans: int = 2        # max replans per tool call
) -> str:
    """
    ReAct loop with replanning.
 
    Flow:
    THOUGHT → ACTION → OBSERVATION
      → if empty: REPLAN → new ACTION → OBSERVATION
      → if still empty after max_replans: honest FINAL ANSWER
      → if result found: FINAL ANSWER
    """
 
    history = history or []
 
    # Last 6 messages for context
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-6:]
    ]
 
    # system prompt + history + current query
    turns = (
        [{"role": "system", "content": build_system_prompt()}]
        + history_messages
        + [{"role": "user", "content": user_query}]
    )
 
    # Track replans per tool to enforce max_replans limit
    replan_counts: dict[str, int] = {}
 
    for iteration in range(max_iterations):
        print(f"\n[Agent] Iteration {iteration + 1}")
 
        # Step 1 — call LLM
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
 
        # Step 2 — reject if LLM mixed ACTION/REPLAN with FINAL ANSWER
        if (has_action or has_replan) and has_final_answer:
            print("[Agent] WARNING: LLM mixed action with final answer — rejecting")
            turns.append({"role": "assistant", "content": llm_output})
            turns.append({
                "role": "user",
                "content": (
                    "You cannot write ACTION or REPLAN and FINAL ANSWER together.\n"
                    "ONE response = ONE action only.\n"
                    "First call the tool. Wait for OBSERVATION. "
                    "Then give FINAL ANSWER in the next step."
                )
            })
            continue
 
        # Step 3 — clean FINAL ANSWER
        if has_final_answer:
            answer = llm_output.split("FINAL ANSWER:")[-1].strip()
            print(f"[Agent] Final answer: {answer[:100]}...")
            return answer
 
        # Step 4 — parse REPLAN (retry with different query)
        replan_match = re.search(r"REPLAN:\s*(\w+)", llm_output)
        if has_replan and replan_match:
            input_match = re.search(r"INPUT:\s*(.+)", llm_output)
            if not input_match:
                turns.append({"role": "assistant", "content": llm_output})
                turns.append({
                    "role": "user",
                    "content": "REPLAN must include INPUT. Please provide a new search query."
                })
                continue
 
            tool_name  = replan_match.group(1).strip()
            tool_input = input_match.group(1).strip()
 
            # Check replan limit
            replan_counts[tool_name] = replan_counts.get(tool_name, 0) + 1
 
            if replan_counts[tool_name] > max_replans:
                print(f"[Agent] Max replans reached for {tool_name} — forcing honest answer")
                turns.append({"role": "assistant", "content": llm_output})
                turns.append({
                    "role": "user",
                    "content": (
                        f"You have tried {max_replans} times with different queries "
                        f"and found no results. "
                        f"Please give an honest FINAL ANSWER telling the customer "
                        f"you couldn't find what they were looking for."
                    )
                })
                continue
 
            print(f"[Agent] REPLAN {replan_counts[tool_name]}/{max_replans} | "
                  f"Tool: {tool_name} | New input: {tool_input}")
 
            # Call tool with new query
            observation = call_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                user_id=user_id
            )
 
            print(f"[Agent] Replan observation: {observation[:200]}...")
 
            turns.append({"role": "assistant", "content": llm_output})
 
            # If still empty — tell LLM to replan again or give up
            if is_empty_result(observation):
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
                    "content": (
                        f"OBSERVATION:\n{observation}\n\n"
                        "Good — results found. Now give your FINAL ANSWER "
                        "based on this observation."
                    )
                })
            continue
 
        # Step 5 — parse regular ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", llm_output)
        input_match  = re.search(r"INPUT:\s*(.+)",   llm_output)
 
        if not action_match or not input_match:
            turns.append({"role": "assistant", "content": llm_output})
            turns.append({
                "role": "user",
                "content": (
                    "Please follow the exact format:\n"
                    "THOUGHT: ...\n"
                    "ACTION: tool_name\n"
                    "INPUT: your query\n\n"
                    "Or after receiving results:\n"
                    "THOUGHT: ...\n"
                    "FINAL ANSWER: your answer"
                )
            })
            continue
 
        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()
 
        print(f"[Agent] Tool: {tool_name} | Input: {tool_input}")
 
        # Step 6 — call tool
        observation = call_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            user_id=user_id
        )
 
        print(f"[Agent] Observation: {observation[:200]}...")
 
        turns.append({"role": "assistant", "content": llm_output})
 
        # Step 7 — if empty result → trigger replanning
        if is_empty_result(observation):
            print(f"[Agent] Empty result detected — triggering replan")
            turns.append({
                "role": "user",
                "content": build_replan_prompt(tool_name, tool_input, 1)
            })
        else:
            # Good result — ask for final answer
            turns.append({
                "role": "user",
                "content": (
                    f"OBSERVATION:\n{observation}\n\n"
                    "Now give your FINAL ANSWER based on this observation. "
                    "Do not call more tools unless absolutely necessary."
                )
            })
 
    return "I wasn't able to find a complete answer. Could you rephrase your question?"