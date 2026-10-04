from .settings import settings
from xgboost import XGBClassifier


def xgboost_model():
    # Fixed modest complexity, no tuning or early stopping on outer test samples.
    return XGBClassifier(**settings()["xgboost"],objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=settings()["random_state"])
