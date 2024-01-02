import numpy as np
import pandas as pd
import sklearn.base
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from StratifiedGroupKFoldRequiresGroups import StratifiedGroupKFoldRequiresGroups
import inspect

from xgboost_label_encoding import (
    XGBoostClassifierWithLabelEncoding,
    XGBoostClassifierWithLabelEncodingWithCV,
    XGBoostCV,
)


@pytest.fixture
def clf_basic():
    return XGBoostClassifierWithLabelEncoding(n_estimators=10, class_weight="balanced")


@pytest.fixture
def clf_cv():
    return XGBoostClassifierWithLabelEncodingWithCV(
        cv=StratifiedKFold(n_splits=2), max_num_trees=20, class_weight="balanced"
    )


# To pass fixtures to parametrize(), we use the lazy_fixture extension.
model_fixtures = [
    pytest.lazy_fixture("clf_basic"),
    pytest.lazy_fixture("clf_cv"),
]


@pytest.mark.parametrize(
    "estimator",
    model_fixtures,
)
def test_sklearn_clonable(estimator):
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
def data_multiclass():
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, n_clusters_per_class=1
    )
    y = pd.Series(y).replace(dict(enumerate(["Covid", "Healthy", "HIV"]))).to_numpy()
    return X, y


@pytest.fixture
def data_binary():
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_clusters_per_class=1
    )
    y = pd.Series(y).replace(dict(enumerate(["Healthy", "HIV"]))).to_numpy()
    return X, y


# To pass fixtures to parametrize(), we use the lazy_fixture extension.
data_fixtures = [
    pytest.lazy_fixture("data_multiclass"),
    pytest.lazy_fixture("data_binary"),
]


@pytest.fixture
def bad_data():
    # This data should cause fitting a model that learns nothing.
    X = pd.DataFrame(np.random.randn(7, 5))
    y = pd.Series(["HIV", "Healthy", "Covid", "Healthy", "Covid", "HIV", "Healthy"])
    return X, y


@pytest.mark.parametrize("data", data_fixtures)
@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_xgboost_label_encoding(data, clf):
    X, y = data
    clf = clf.fit(X, y)
    unique_classes = np.unique(y)
    assert np.array_equal(clf.classes_, unique_classes)
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), len(unique_classes))
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))
    # Confirm again that cloning works, even after a real fit
    sklearn.base.clone(clf)


@pytest.mark.parametrize("data", data_fixtures)
@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_has_other_sklearn_properties(data, clf):
    X, y = data
    # set feature names
    X = pd.DataFrame(X)
    X = X.rename(columns=lambda s: f"col{s}")
    assert np.array_equal(X.columns, ["col0", "col1", "col2", "col3", "col4"])

    # Fit without feature names first
    clf = clf.fit(X.values, y)
    assert clf.n_features_in_ == X.shape[1]
    assert not hasattr(clf, "feature_names_in_")

    # Fit with feature names
    clf = clf.fit(X, y)
    assert clf.n_features_in_ == X.shape[1]
    assert np.array_equal(clf.feature_names_in_, X.columns)

    assert clf.feature_importances_.shape == (X.shape[1],)


@pytest.mark.parametrize("data", data_fixtures)
@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_fit_without_specifying_objective(data, clf):
    """Sanity check that we don't need to set binary or multiclass objective manually. Relies on model_fixtures not having objective pre-set."""
    X, y = data
    clf = clf.fit(X, y)
    unique_classes = np.unique(y)
    assert np.array_equal(clf.classes_, unique_classes)
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), len(unique_classes))
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))


@pytest.mark.parametrize("data", data_fixtures)
@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_class_weight_parameter_hidden_from_inner_xgboost(data, clf):
    """
    Confirm that class_weight is not passed to inner xgboost
    Otherwise, we'd get this warning from calling fit():
    WARNING: xgboost/src/learner.cc:767:
    Parameters: { "class_weight" } are not used.

    Relies on model_fixtures having class_weight="balanced" set.
    """
    X, y = data

    clf = clf.fit(X, y)

    assert clf.class_weight == "balanced"
    assert "class_weight" not in clf.get_xgb_params()
    assert "_original_feature_names_" not in clf.get_xgb_params()
    assert "_transformed_feature_names_" not in clf.get_xgb_params()

    # Confirm again after cloning
    clf = sklearn.base.clone(clf)
    clf = clf.fit(X, y)
    assert clf.class_weight == "balanced"
    assert "class_weight" not in clf.get_xgb_params()


@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_fit_with_illegal_feature_names(clf):
    # Create a simple classification problem
    # Include forbidden characters in column names
    # "feature_names must be string, and may not contain [, ] or <"
    original_column_names = ["col[0]", "col]1", "col<2", "col3"]
    X, y = make_classification(n_samples=100, n_features=len(original_column_names))
    X = pd.DataFrame(X, columns=original_column_names)
    assert np.array_equal(X.columns, original_column_names)

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


@pytest.mark.parametrize(
    "clf",
    model_fixtures,
)
def test_unique_renaming_of_columns(
    clf,
):
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


@pytest.mark.parametrize("data", data_fixtures)
@pytest.mark.parametrize(
    ["clf", "expected_to_call_xgboostcv_fit"],
    [(pytest.lazy_fixture("clf_cv"), True), (pytest.lazy_fixture("clf_basic"), False)],
)
def test_cv_module_resolution_order(
    clf, expected_to_call_xgboostcv_fit: bool, data, mocker
):
    """
    Make sure that calling XGBoostClassifierWithLabelEncodingWithCV.fit() calls XGBoostCV.fit() first, not XGBClassifier.fit() directly
    This is controlled by the MRO (method resolution order) of the class.
    """
    # We want to check that XGBoostCV.fit() is called first.
    # Instead of mocker.patch.object(XGBoostCV, "fit" followed by XGBoostCV.fit.assert_called(),
    # use a spy to preserve the functionality of the method, so that rest of XGBoostClassifierWithLabelEncodingWithCV.fit() works.
    spy = mocker.spy(XGBoostCV, "fit")

    clf.fit(*data)

    if expected_to_call_xgboostcv_fit:
        # Check that XGBoostCV's fit method was called, which means it was called first rather than XGBClassifier's fit method being called directly
        spy.assert_called()
    else:
        # Also sanity check that XGBoostCV's fit method was not called for clf_basic
        spy.assert_not_called()


@pytest.mark.parametrize("data", data_fixtures)
def test_cv(clf_cv, data):
    # Confirm CV is happening
    X, y = data

    clf_cv.max_num_trees = 200

    # Fit the model
    clf_cv.fit(X, y)

    # Early stopping would have stopped the training before 200 estimators
    assert clf_cv.n_estimators < 200
    # And shouldn't be 100, the default number of estimators
    assert clf_cv.n_estimators != 100


# @pytest.mark.parametrize("clf", model_fixtures)
# Not able to get this to fail with clf_basic weirdly, even if setting clf.n_estimators = 1
def test_fit_fails_if_feature_importances_are_all_zero(bad_data, clf_cv):
    X, y = bad_data
    with pytest.raises(ValueError, match="All feature importances are zero"):
        clf_cv.fit(X, y)


@pytest.mark.parametrize("data", data_fixtures)
def test_cv_groups_parameter_passed_through(data):
    # use StratifiedGroupKFoldRequiresGroups instead of StratifiedKFold to ensure groups parameter goes through
    # if groups not provided, it will raise ValueError: StratifiedGroupKFoldRequiresGroups requires groups argument to be provided to split().
    cv = StratifiedGroupKFoldRequiresGroups(n_splits=2)
    clf = XGBoostClassifierWithLabelEncodingWithCV(
        cv=cv, max_num_trees=20, class_weight="balanced"
    )
    X, y = data
    groups = np.random.choice([0, 1], size=X.shape[0])

    # before proceeding, confirm that fit() method signature reveals support for weights and groups
    fit_parameters = inspect.signature(clf.fit).parameters
    assert "sample_weight" in fit_parameters.keys()
    assert "groups" in fit_parameters.keys()

    clf.fit(X, y, groups=groups)


def test_basic_classifier_does_not_advertise_groups_parameter(clf_basic):
    # follow up to test_cv_groups_parameter_passed_through:
    # confirm that basic classifier does not advertise groups parameter
    fit_parameters = inspect.signature(clf_basic.fit).parameters
    assert "sample_weight" in fit_parameters.keys()
    assert "groups" not in fit_parameters.keys()
