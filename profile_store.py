# profile_store.py
# Long-term memory — stores ONLY purchase history
# Works for ANY merchant — not specific to phone cases

import os
import json
import time
from config import BASE_DIR
from pathlib import Path

PROFILE_DIR = str(Path(BASE_DIR) / "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DEFAULT — only purchases, nothing else
# ─────────────────────────────────────────────
def default_profile() -> dict:
    return {
        "purchases": []
    }


# ─────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────
def get_profile_path(user_id: str) -> str:
    return os.path.join(PROFILE_DIR, f"{user_id}.json")


def load_profile(user_id: str) -> dict:
    path = get_profile_path(user_id)

    if not os.path.exists(path):
        return default_profile()

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Safety check
        if "purchases" not in data:
            data["purchases"] = []

        return data

    except Exception as e:
        print(f"[PROFILE LOAD ERROR] {e}")
        return default_profile()


def save_profile(user_id: str, profile: dict) -> None:
    path = get_profile_path(user_id)

    try:
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)

    except Exception as e:
        print(f"[PROFILE SAVE ERROR] {e}")


# ─────────────────────────────────────────────
# WRITE HELPER
# ─────────────────────────────────────────────
def record_purchase(profile: dict, product: str, quantity: int, order_id: str) -> None:
    """
    Call this when an order is placed.
    Saves what the customer bought for future reference.
    """
    profile["purchases"].append({
        "date":     time.strftime("%Y-%m-%d"),
        "product":  product,
        "quantity": quantity,
        "order_id": order_id
    })
    print(f"[Profile] Purchase recorded: {product} x{quantity}")


# ─────────────────────────────────────────────
# READ HELPER — used by agent
# ─────────────────────────────────────────────
def needs_purchase_context(query: str) -> bool:
    """
    Decides if the customer's query is about something they previously bought.
    Only inject purchase history when this returns True.
    """
    PAST_PURCHASE_SIGNALS = [
        "i bought", "i ordered", "my order",
        "last time", "previously", "i purchased",
        "return", "refund", "exchange",
        "the one i got", "i got it",
        "my purchase", "tracking", "order status",
        "wrong item", "damaged", "not working"
    ]
    print(f"[Profile] Checking if query needs purchase context: '{query}'")
    query_lower = query.lower()
    return any(signal in query_lower for signal in PAST_PURCHASE_SIGNALS)


def get_purchase_context(profile: dict) -> str:
    """
    Builds a context string from purchase history.
    Only called when needs_purchase_context() is True.
    Returns empty string if no purchases yet.
    """
    purchases = profile.get("purchases", [])

    if not purchases:
        return ""

    # Show last 5 purchases only
    recent = purchases[-5:]

    lines = ["Customer's recent purchases:"]
    for p in recent:
        lines.append(
            f"- {p['date']}: {p['product']} x{p['quantity']} (Order: {p['order_id']})"
        )

    return "\n".join(lines)