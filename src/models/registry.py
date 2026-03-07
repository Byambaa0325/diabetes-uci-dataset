"""Central model registry — maps model name → {cls, framework}.

To add a new model:
  - PyTorch: define an nn.Module in src/networks/, add an entry with framework="torch"
  - Sklearn:  add an entry with framework="sklearn" — no extra file needed
"""
from src.networks.mlp import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

MODEL_REGISTRY: dict[str, dict] = {
    # --- PyTorch ---
    "mlp": {
        "cls": MLP,
        "framework": "torch",
    },
    # --- Sklearn ---
    "logistic_regression": {
        "cls": LogisticRegression,
        "framework": "sklearn",
    },
    "random_forest": {
        "cls": RandomForestClassifier,
        "framework": "sklearn",
    },
    "gradient_boosting": {
        "cls": GradientBoostingClassifier,
        "framework": "sklearn",
    },
    "svm": {
        "cls": SVC,
        "framework": "sklearn",
    },
}
