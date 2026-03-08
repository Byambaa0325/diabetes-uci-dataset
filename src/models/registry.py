"""Central model registry — maps model name → {cls, framework}.

To add a new model:
  - PyTorch: define an nn.Module in src/networks/, add an entry with framework="torch"
  - Sklearn:  add an entry with framework="sklearn" — no extra file needed
  - Optional deps: use the try/except pattern at the bottom (see tabicl)
"""
from src.networks.mlp import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC

try:
    from tabicl import TabICLClassifier as _TabICL
    _TABICL_AVAILABLE = True
except ImportError:
    _TabICL = None
    _TABICL_AVAILABLE = False

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
    "linear_svm": {
        "cls": LinearSVC,
        "framework": "sklearn",
    },
}

# --- Optional: foundation models ---
if _TABICL_AVAILABLE:
    MODEL_REGISTRY["tabicl"] = {
        "cls":       _TabICL,
        "framework": "sklearn",
    }
