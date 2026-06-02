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
 
When you have enough information to answer:
THOUGHT: I have enough information to answer.
FINAL ANSWER: <your friendly, clear answer>
 
Rules:
- Always write THOUGHT before ACTION or FINAL ANSWER
- Never guess product prices or policy details — always use a tool first
- If question is about both product AND policy, call both tools one at a time
- Base your FINAL ANSWER only on what tools returned
- Keep FINAL ANSWER short, friendly, and helpful
- If nothing found, politely say so
- ONE RESPONSE = ONE ACTION or ONE FINAL ANSWER — never both in the same response
- You MUST wait for the OBSERVATION before writing FINAL ANSWER
"""


def run_react_agent(
    user_query: str,
    user_id: str = "agent",
    history: list = None,
    max_iterations: int = 6
) -> str:

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
        has_final_answer = "FINAL ANSWER:" in llm_output

        # Step 2 — LLM wrote both ACTION and FINAL ANSWER in same response
        # This means it hallucinated without waiting for observation
        # Reject and force it to do one thing at a time
        if has_action and has_final_answer:
            print("[Agent] WARNING: LLM wrote ACTION and FINAL ANSWER together — rejecting")
            turns.append({"role": "assistant", "content": llm_output})
            turns.append({
                "role": "user",
                "content": (
                    "You cannot write ACTION and FINAL ANSWER in the same response.\n"
                    "ONE response = ONE action only.\n"
                    "First call the tool. Wait for OBSERVATION. "
                    "Then in the next step give your FINAL ANSWER.\n\n"
                    "Start again — write only the ACTION and INPUT now."
                )
            })
            continue

        # Step 3 — clean FINAL ANSWER
        if has_final_answer:
            return llm_output.split("FINAL ANSWER:")[-1].strip()

        # Step 3 — parse ACTION and INPUT
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
                    "Or:\n"
                    "THOUGHT: ...\n"
                    "FINAL ANSWER: your answer"
                )
            })
            continue

        tool_name  = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        print(f"[Agent] Tool: {tool_name} | Input: {tool_input}")

        # Step 4 — call tool
        observation = call_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            user_id=user_id
        )

        print(f"[Agent] Observation: {observation[:200]}...")

        # Step 5 — feed observation back
        turns.append({"role": "assistant", "content": llm_output})
        turns.append({
            "role": "user",
            "content": (
                f"OBSERVATION:\n{observation}\n\n"
                "Continue. Call another tool if still needed, "
                "or give FINAL ANSWER if you have enough information."
            )
        })

    return "I wasn't able to find a complete answer. Could you rephrase your question?"