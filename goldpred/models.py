"""Candidate classifiers, including a naive baseline every model must beat."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = [
    "Majority class (baseline)",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]


def make_model(name: str, random_state: int = 42):
    """Return a fresh, unfitted estimator for `name`."""
    if name == "Majority class (baseline)":
        # Always predicts the most common training direction; probabilities = class prior.
        return DummyClassifier(strategy="prior")
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    if name == "Random Forest":
        # Large leaves keep the forest from memorising noise in daily returns.
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=random_state
        )
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=random_state
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
