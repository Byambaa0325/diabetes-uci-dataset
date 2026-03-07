"""Training entry point.

Can be called as a function from notebooks:
    from scripts.train import run_training
    run_training(config)

Or from the CLI:
    python scripts/train.py '{"model": "logistic_regression", ...}'
"""
import json
import sys
from pathlib import Path

# Ensure repo root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.trainers import run_training  # noqa: E402 — path setup must come first

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py '<config_json>'")
        sys.exit(1)
    config = json.loads(sys.argv[1])
    run_training(config)
