import numpy as np
import pandas as pd
import sklearn.base
import pytest

from xgboost_label_encoding import (
    XGBoostClassifierWithLabelEncoding,
)


def test_sklearn_clonable():
    estimator = XGBoostClassifierWithLabelEncoding()
    # Check that supports cloning with sklearn.base.clone
    estimator_clone = sklearn.base.clone(estimator)

    # not fitted yet
    assert not hasattr(estimator, "label_encoder_")
    assert not hasattr(estimator_clone, "label_encoder_")
    assert not hasattr(estimator, "classes_")
    assert not hasattr(estimator_clone, "classes_")

    # pretend it is fitted
    estimator.classes_ = np.array(["a", "b"])
    assert hasattr(estimator, "classes_")

    # confirm clone is not fitted
    estimator_clone_2 = sklearn.base.clone(estimator)
    assert not hasattr(estimator_clone_2, "classes_")


@pytest.fixture
def data():
    X = pd.DataFrame(np.random.randn(5, 5))
    y = pd.Series(["HIV", "Healthy", "Covid", "Healthy", "Covid"])
    return X, y


@pytest.fixture
def data_binary():
    X = pd.DataFrame(np.random.randn(5, 5))
    y = pd.Series(["HIV", "Healthy", "HIV", "Healthy", "HIV"])
    return X, y


def test_xgboost_label_encoding(data):
    X, y = data
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
        objective="multi:softprob",
    ).fit(X, y)
    assert np.array_equal(clf.classes_, ["Covid", "HIV", "Healthy"])
    assert clf.predict(X).shape == (5,)
    assert clf.predict_proba(X).shape == (5, 3)
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))
    # Confirm again that cloning works, even after a real fit
    sklearn.base.clone(clf)


def test_has_other_sklearn_properties(data):
    X, y = data
    # set feature names
    X = X.rename(columns=lambda s: f"col{s}")

    # Fit without feature names first
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X.values, y)
    assert clf.n_features_in_ == 5
    assert not hasattr(clf, "feature_names_in_")

    # Fit with feature names
    clf = clf.fit(X, y)
    assert clf.n_features_in_ == 5
    assert np.array_equal(
        clf.feature_names_in_, ["col0", "col1", "col2", "col3", "col4"]
    )

    assert clf.feature_importances_.shape == (5,)


# Sanity check that we don't need to set objective
def test_fit_multiclass_without_specifying_objective(data):
    X, y = data
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X, y)
    assert np.array_equal(clf.classes_, ["Covid", "HIV", "Healthy"])
    assert clf.predict(X).shape == (5,)
    assert clf.predict_proba(X).shape == (5, 3)
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))


def test_fit_binary_without_specifying_objective(data_binary):
    X_binary, y_binary = data_binary
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X_binary, y_binary)
    assert np.array_equal(clf.classes_, ["HIV", "Healthy"])
    assert clf.predict(X_binary).shape == (5,)
    assert clf.predict_proba(X_binary).shape == (5, 2)
    assert all(
        predicted_label in clf.classes_ for predicted_label in clf.predict(X_binary)
    )
