from pathlib import Path

# Path to the policies JSON file — imported as `policies` elsewhere.
policies = str(Path(__file__).resolve().parent / "policies.json")

__all__ = ["policies"]
