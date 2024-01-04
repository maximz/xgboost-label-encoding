"""Test XGBoostCV on its own."""

import numpy as np
from sklearn.datasets import make_classification
from xgboost_label_encoding import XGBoostCV
import pytest
from StratifiedGroupKFoldRequiresGroups import StratifiedGroupKFoldRequiresGroups
import inspect


@pytest.mark.parametrize("num_classes", [2, 3])
def test_cv(num_classes: int):
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=num_classes, n_clusters_per_class=1
    )
    groups = np.random.choice([0, 1], size=100)

    # use StratifiedGroupKFoldRequiresGroups instead of StratifiedKFold to ensure groups parameter goes through
    # if groups not provided, it will raise ValueError: StratifiedGroupKFoldRequiresGroups requires groups argument to be provided to split().
    cv = StratifiedGroupKFoldRequiresGroups(n_splits=2, random_state=42, shuffle=True)

    # Initialize the classifier
    clf = XGBoostCV(cv=cv)

    # before proceeding, confirm that fit() method signature reveals support for weights and groups
    fit_parameters = inspect.signature(clf.fit).parameters
    assert "sample_weight" in fit_parameters.keys()
    assert "groups" in fit_parameters.keys()

    # Fit the model
    clf.fit(X, y, groups=groups)

    # Early stopping would have stopped the training before 200 estimators
    assert clf.n_estimators < 200

    # Confirm our new parameters are not exposed as unnecessary parameters to inner xgboost
    assert "max_num_trees" not in clf.get_xgb_params()
