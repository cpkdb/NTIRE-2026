from sklearn.metrics import roc_auc_score
import numpy as np


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC score."""
    return roc_auc_score(labels, scores)
