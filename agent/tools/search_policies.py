from agent.data import policies
import json

with open(policies,"r") as f:
    POLICIES = json.load(f)


def search_policies(query: str) -> str:
    """
    Tool: search_policies
    Searches store policies based on the customer query.
    Returns the relevant policy content as a string.
 
    Use when customer asks about:
    - Shipping or delivery times
    - Return or refund process
    - How to return a product
    - When order will arrive
    - Money back or damaged product
    """
    query_lower = query.lower()
    matched_policies = []

    for policy_key, policy_data in POLICIES.items():
        # Check if any keyword matches the query
        for keyword in policy_data['keywords']:
            if keyword in query_lower:
                matched_policies.append(
                    f"{policy_data['title']}:\n{policy_data['content']}"
                )
                break  # avoid adding same policy twice

    if not matched_policies:
        # No keyword matched — return all policies
        all_policies = []
        for policy_key, policy_data in POLICIES.items():
            all_policies.append(
                f"{policy_data['title']}:\n{policy_data['content']}"
            )
        return "\n\n".join(all_policies)
 
    return "\n\n".join(matched_policies)