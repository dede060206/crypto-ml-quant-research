from .settings import settings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def baseline_logreg():
    # Standardization is fitted afresh ONLY on each fold's training X.
    return make_pipeline(StandardScaler(),LogisticRegression(**settings()["logreg"],random_state=settings()["random_state"]))
