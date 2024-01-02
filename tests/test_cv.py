"""Test XGBoostCV on its own."""

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from xgboost_label_encoding import XGBoostCV
import pytest


@pytest.mark.parametrize("num_classes", [2, 3])
def test_cv(num_classes: int):
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=num_classes, n_clusters_per_class=1
    )

    # Initialize the classifier
    clf = XGBoostCV(cv=StratifiedKFold())

    # Fit the model
    clf.fit(X, y)

    # Early stopping would have stopped the training before 200 estimators
    assert clf.n_estimators < 200

    # Confirm our new parameters are not exposed as unnecessary parameters to inner xgboost
    assert "max_num_trees" not in clf.get_xgb_params()
