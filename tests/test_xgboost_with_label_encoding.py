import numpy as np
import pandas as pd
import sklearn.base
import pytest
from sklearn.datasets import make_classification

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
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), 3)
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))
    # Confirm again that cloning works, even after a real fit
    sklearn.base.clone(clf)


def test_has_other_sklearn_properties(data):
    X, y = data
    # set feature names
    X = X.rename(columns=lambda s: f"col{s}")
    assert np.array_equal(X.columns, ["col0", "col1", "col2", "col3", "col4"])

    # Fit without feature names first
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X.values, y)
    assert clf.n_features_in_ == X.shape[1]
    assert not hasattr(clf, "feature_names_in_")

    # Fit with feature names
    clf = clf.fit(X, y)
    assert clf.n_features_in_ == X.shape[1]
    assert np.array_equal(clf.feature_names_in_, X.columns)

    assert clf.feature_importances_.shape == (X.shape[1],)


# Sanity check that we don't need to set objective
def test_fit_multiclass_without_specifying_objective(data):
    X, y = data
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X, y)
    assert np.array_equal(clf.classes_, ["Covid", "HIV", "Healthy"])
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), 3)
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))


def test_fit_binary_without_specifying_objective(data_binary):
    X_binary, y_binary = data_binary
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
    ).fit(X_binary, y_binary)
    assert np.array_equal(clf.classes_, ["HIV", "Healthy"])
    assert clf.predict(X_binary).shape == (len(y_binary),)
    assert clf.predict_proba(X_binary).shape == (len(y_binary), 2)
    assert all(
        predicted_label in clf.classes_ for predicted_label in clf.predict(X_binary)
    )


def test_class_weight_parameter_hidden_from_inner_xgboost(data):
    X, y = data

    # Confirm that class_weight is not passed to inner xgboost
    # Otherwise, we'd get this warning from calling fit():
    # WARNING: xgboost/src/learner.cc:767:
    # Parameters: { "class_weight" } are not used.
    clf = XGBoostClassifierWithLabelEncoding(
        n_estimators=10,
        class_weight="balanced",
    ).fit(X, y)

    assert clf.class_weight == "balanced"
    assert "class_weight" not in clf.get_xgb_params()
    assert "_original_feature_names_" not in clf.get_xgb_params()
    assert "_transformed_feature_names_" not in clf.get_xgb_params()

    # Confirm again after cloning
    clf = sklearn.base.clone(clf)
    clf = clf.fit(X, y)
    assert clf.class_weight == "balanced"
    assert "class_weight" not in clf.get_xgb_params()


def test_fit_with_illegal_feature_names():
    # Create a simple classification problem
    # Include forbidden characters in column names
    # "feature_names must be string, and may not contain [, ] or <"
    original_column_names = ["col[0]", "col]1", "col<2", "col3"]
    X, y = make_classification(n_samples=100, n_features=len(original_column_names))
    X = pd.DataFrame(X, columns=original_column_names)
    assert np.array_equal(X.columns, original_column_names)

    # Initialize the classifier
    clf = XGBoostClassifierWithLabelEncoding()

    # Fit the model
    clf.fit(X, y)

    # Check that the original feature names are preserved
    assert np.array_equal(
        clf.feature_names_in_, X.columns
    ), "Original feature names are not preserved correctly."

    # Check that the renaming occurred in the inner fitted model: access the feature names from the inner model
    inner_model_feature_names = clf.get_booster().feature_names
    assert np.array_equal(clf._transformed_feature_names_, inner_model_feature_names)
    for transformed in inner_model_feature_names:
        assert (
            "[" not in transformed and "," not in transformed and "<" not in transformed
        ), "Forbidden character found in transformed feature names."

    # Check that predict functions work with the original feature names
    predictions = clf.predict(X)
    assert len(predictions) == len(
        y
    ), "Predict function does not return the correct number of predictions."
    prob_predictions = clf.predict_proba(X)
    assert prob_predictions.shape == (
        len(y),
        2,
    ), "Predict_proba function does not return probabilities in expected shape."
    assert np.array_equal(
        clf._transform_input(X).columns, inner_model_feature_names
    ), "Transformed feature names do not match the inner model."


def test_unique_renaming_of_columns():
    # Create a DataFrame where columns would have the same name after forbidden character removal
    # Renaming replaces [, ] and < with _
    cols = [
        "col[0]",
        "col[0<",
        "col[0[]",
        "col<0>",
        "col0",
        "col_0_",
        "col_0_1",
        "col_0__1",
        "col_0___1",
    ]
    X, y = make_classification(n_samples=100, n_features=len(cols))
    X = pd.DataFrame(X, columns=cols)
    clf = XGBoostClassifierWithLabelEncoding()

    # Fit the classifier
    clf.fit(X, y)

    # Retrieve the transformed feature names from the inner model (see xgboost original implementation)
    transformed_feature_names = clf.get_booster().feature_names

    # Check that all transformed feature names are unique
    assert len(set(transformed_feature_names)) == len(
        transformed_feature_names
    ), "Transformed feature names are not unique."
    assert np.array_equal(
        transformed_feature_names,
        [
            "col_0__1_1",
            "col_0__1_1_1",
            "col_0__",
            "col_0>",
            "col0",
            "col_0_",
            "col_0_1",
            "col_0__1",
            "col_0___1",
        ],
    ), transformed_feature_names[-2:]

    # Ensure the original names are still intact
    assert np.array_equal(
        clf.feature_names_in_, X.columns
    ), "The preserved feature names do not match the original."
