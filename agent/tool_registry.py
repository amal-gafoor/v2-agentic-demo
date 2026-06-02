# agent/tool_registry.py

from agent.tools.search_products import search_products
from agent.tools.search_policies import search_policies


TOOL_REGISTRY = {
    "search_products": {
        "function": search_products,
        "description": (
            "Search for products by name, features, price, or category. "
            "Use when customer asks about product details, pricing, "
            "availability, specs, or wants to compare products."
        ),
    },
    "search_policies": {
        "function": search_policies,
        "description": (
            "Search store policies. "
            "Use when customer asks about shipping, delivery time, "
            "returns, refunds, or damaged products."
        ),
    }
}


def call_tool(tool_name: str, tool_input: str, user_id: str = "agent") -> str:
    """
    Calls a tool by name and returns result as string.
    """
    if tool_name not in TOOL_REGISTRY:
        available = list(TOOL_REGISTRY.keys())
        return f"Error: tool '{tool_name}' not found. Available: {available}"

    try:
        if tool_name == "search_products":
            return search_products(tool_input, user_id=user_id)

        elif tool_name == "search_policies":
            return search_policies(tool_input)

    except Exception as e:
        return f"Tool error in '{tool_name}': {str(e)}"