from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def build_confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)
