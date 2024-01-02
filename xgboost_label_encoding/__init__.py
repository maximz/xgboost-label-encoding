"""Xgboost Label Encoding."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.6"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict, Optional, Tuple, Union
from typing_extensions import Self
import sklearn.utils.class_weight
import xgboost as xgb
from joblib import Parallel, delayed
import inspect
import sklearn.model_selection


class XGBoostClassifierWithLabelEncoding(xgb.XGBClassifier):
    """
    Wrapper around XGBoost XGBClassifier with label encoding for the target y label.

    Native XGBoost doesn't support string labels, and XGBClassifier's `use_label_encoder` property was removed in 1.6.0.
    Unfortunately, sklearn's `LabelEncoder` for `y` target values does not play well with sklearn pipelines.

    Our workaround: wrap XGBClassifier in this wrapper for automatic label encoding of y.
    Use this in place of XGBClassifier, and `y` will automatically be label encoded.

    Additional features:
    - Automatic class weight rebalancing as in sklearn
    - Automatic renaming of column names passed through to xgboost to avoid xgboost error: "feature_names must be string, and may not contain [, ] or <"
    - Automatic rejection of the fitted model if all feature importances are zero. This is a sign that the model did not learn anything. Consider changing the hyperparameters. (Set fail_if_nothing_learned=False to disable)
    """

    _original_feature_names_: Optional[np.ndarray]
    _transformed_feature_names_: Optional[np.ndarray]

    def __init__(
        self,
        class_weight: Optional[Union[dict, str]] = None,
        fail_if_nothing_learned: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_weight = class_weight
        # fail_if_nothing_learned: Rejects the final model if all feature importances are zero. This is a sign that the model did not learn anything. Consider changing the hyperparameters.
        self.fail_if_nothing_learned = fail_if_nothing_learned

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Self:
        if self.class_weight is not None:
            # Use sklearn to compute class weights, then map to individual sample weights
            sample_weight_computed = sklearn.utils.class_weight.compute_sample_weight(
                class_weight=self.class_weight, y=y
            )
            if sample_weight is None:
                # No sample weights were provided. Just use the ones derived from class weights.
                sample_weight = sample_weight_computed
            else:
                # Sample weights were already provided. We need to combine with class-derived weights.
                # First, confirm shape matches
                if sample_weight.shape[0] != sample_weight_computed.shape[0]:
                    raise ValueError(
                        "Provided sample_weight has different number of samples than y."
                    )
                # Then, multiply the two
                sample_weight = sample_weight * sample_weight_computed

        # Encode y labels
        self.label_encoder_ = LabelEncoder()
        transformed_y = self.label_encoder_.fit_transform(y)

        if len(self.label_encoder_.classes_) < 2:
            raise ValueError(
                f"Training data needs to have at least 2 classes, but the data contains only one class: {self.label_encoder_.classes_[0]}"
            )

        # Store original column names. Xgboost will see cleaned-up versions but this property will expose any original illegal column names.
        # Initialize as None in case we have no column names
        self._original_feature_names_ = None
        self._transformed_feature_names_ = None

        # Store column names if X is a pandas DataFrame, and rename as necessary
        if isinstance(X, pd.DataFrame):
            # Avoid error: "feature_names must be string, and may not contain [, ] or <"

            # Renaming columns if X is a pandas DataFrame and contains forbidden characters
            forbidden_chars = "[]<"

            # Store original names
            self._original_feature_names_ = X.columns.to_numpy()

            # Ensure all column names are strings
            X.columns = X.columns.map(str)

            # Track new column names as we define them.
            # Initialize with existing column names to avoid changing those unless needed
            new_column_names = set(X.columns)

            # Store rename mapping
            column_mapping = {}

            for col in X.columns:
                new_name = re.sub(f"[{re.escape(forbidden_chars)}]", "_", col)
                if new_name == col:
                    # No renaming happened. Skip to next column.
                    continue

                # Rename is needed. Add to mapping.
                # First, we need to make sure the new column name is unique.
                # If it's not, we'll append "_1" until it is.
                while new_name in new_column_names:
                    # Iterate until we ensure uniqueness
                    new_name += "_1"
                new_column_names.add(new_name)
                column_mapping[col] = new_name

            # Execute renames
            X = X.rename(columns=column_mapping)

            # Store original names
            self._transformed_feature_names_ = X.columns.to_numpy()

        # fit as usual
        super().fit(X, transformed_y, sample_weight=sample_weight, **kwargs)

        # Reject the model if all feature importances are zero.
        if self.fail_if_nothing_learned and np.allclose(self.feature_importances_, 0):
            raise ValueError(
                "All feature importances are zero. This is a sign that the model did not learn anything. Consider changing the hyperparameters."
            )

        # set classes_
        self.classes_: np.ndarray = self.label_encoder_.classes_
        return self

    def _transform_input(self, X: Union[np.ndarray, pd.DataFrame]):
        """Apply the same column renaming logic to the input X as was applied during fit."""
        if isinstance(X, pd.DataFrame):
            if (
                self._original_feature_names_ is None
                or self._transformed_feature_names_ is None
            ):
                raise ValueError(
                    "fit() must be called before predict() or predict_proba()."
                )
            # Convert column names to strings and apply renaming logic
            transformed_cols = {
                old: new
                for old, new in zip(
                    self._original_feature_names_, self._transformed_feature_names_
                )
            }
            X = X.rename(columns=transformed_cols)
        return X

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X = self._transform_input(X)  # Apply the renaming
        return super().predict_proba(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X = self._transform_input(X)  # Apply the renaming
        return self.label_encoder_.inverse_transform(super().predict(X))

    @property
    def feature_names_in_(self) -> np.ndarray:
        """Names of features seen during :py:meth:`fit`.
        Defined only when `X` has feature names that are all strings.
        Overriden in our implementation to support fitting with column names that include forbidden characters. Xgboost will see cleaned-up versions but this property will expose the original illegal column names.
        """
        if self._original_feature_names_ is None:
            # This is the error thrown by xgboost's original implementation in this situation:
            raise AttributeError(
                "`feature_names_in_` is defined only when `X` has feature names that are all strings."
            )
        return self._original_feature_names_  # numpy array

    def get_xgb_params(self) -> Dict[str, Any]:
        """
        Get xgboost-specific parameters to be passed into the underlying xgboost C++ code.
        Override the default get_xgb_params() implementation to exclude our wrapper's class_weight parameter from being passed through into xgboost core.

        This avoids the following warning from xgboost:
        WARNING: xgboost/src/learner.cc:767:
        Parameters: { "class_weight" } are not used.
        """
        # Original implementation: https://github.com/dmlc/xgboost/blob/d4d7097accc4db7d50fdc2b71b643925db6bc424/python-package/xgboost/sklearn.py#L795-L816
        params = super().get_xgb_params()

        params_to_remove = [
            "class_weight",
            "fail_if_nothing_learned",
        ]
        for param in params_to_remove:
            if param in params:  # it should be
                # Drop from params
                del params[param]

        return params


class XGBoostCV(xgb.XGBClassifier):
    def __init__(
        self,
        cv: sklearn.model_selection.BaseCrossValidator,
        param_grid: Optional[dict] = None,
        max_num_trees: int = 200,
        # Translates to early_stopping_rounds, but distinct name to not confuse the final post-CV XGBClassifier into thinking we want early stopping.
        # If we name this early_stopping_rounds, we get the following error: AssertionError: Must have at least 1 validation dataset for early stopping.
        early_stopping_patience: int = 10,
        n_jobs: int = 1,
        # Specify metric and objective for binary and multiclass settings.
        metric_binary: str = "logloss",
        metric_multiclass: str = "mlogloss",
        objective_binary: str = "binary:logistic",
        objective_multiclass: str = "multi:softprob",
        # Specify whether lower score is considered better for the chosen metric.
        metric_lower_is_better: bool = True,
        **kwargs,
    ):
        """
        Run cross-validation to choose the best hyperparameters for an XGBoost model.

        Parameters:

        - cv
            Use this cross validation splitter. Groups argument will be fed through.

        - param_grid
            List of hyperparameter dicts to try.
            Make sure the parameter names are valid for both xgboost's native API and xgboost's sklearn API.
            Do not include number of trees (called "num_boost_round" in native API, aka "n_estimators" for xgboost's sklearn API)
            Do not include early_stopping_rounds either.
            If param_grid is not specified, by default we will try a small set of learning rates and min_child_weights.
            See https://xgboost.readthedocs.io/en/release_1.7.0/parameter.html

        - max_num_trees
            Look for best number of trees (aka n_estimators and number of xgboost iterations).
            The final n_estimators will be below this number.

        - early_stopping_patience
            Used as early_stopping_rounds parameter to xgboost.cv().
            Require this many rounds of no improvement before stopping:

            Note: this value is not used directly in the final model fit operation.
            We instead use the best number of boosting rounds found by early stopping.
            To be more precise: After early stopping in each fold during CV operation, we go back to the best iteration across folds and use that number of iterations for final model.

        - n_jobs
            How many hyperparameter settings to try at once in parallel.
            Each hyperparameter setting is launched as a joblib.Parallel job that calls xgboost.cv() for multi-fold cross validation with those chosen parameters.

        - metric_binary, metric_multiclass, metric_lower_is_better
            Configures the metric to use for cross validation and early stopping.

        - objective_binary, objective_multiclass
            Configures the objective to use for xgboost.cv() fits.
        """
        super().__init__(**kwargs)
        # Make sure to register these parameters for removal in get_xgb_params()
        self.cv = cv
        self.param_grid = param_grid
        self.max_num_trees = max_num_trees
        self.early_stopping_patience = early_stopping_patience
        self.n_jobs = n_jobs
        self.metric_binary = metric_binary
        self.metric_multiclass = metric_multiclass
        self.objective_binary = objective_binary
        self.objective_multiclass = objective_multiclass
        self.metric_lower_is_better = metric_lower_is_better

    def get_xgb_params(self) -> Dict[str, Any]:
        """
        Get xgboost-specific parameters to be passed into the underlying xgboost C++ code.
        Override the default get_xgb_params() implementation to exclude our wrapper's class_weight parameter from being passed through into xgboost core.
        """
        # Original implementation: https://github.com/dmlc/xgboost/blob/d4d7097accc4db7d50fdc2b71b643925db6bc424/python-package/xgboost/sklearn.py#L795-L816
        params = super().get_xgb_params()

        params_to_remove = [
            "cv",
            "param_grid",
            "max_num_trees",
            "early_stopping_patience",
            "metric_binary",
            "metric_multiclass",
            "objective_binary",
            "objective_multiclass",
            "metric_lower_is_better",
        ]
        for param in params_to_remove:
            if param in params:
                # Drop from params
                del params[param]

        return params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Fit with cross validation.
        """

        # Choose hyperparameter tuning grid
        param_grid = self.param_grid
        if param_grid is None:
            # Configure small default hyperparameter tuning grid.
            # Do this here for two reasons:
            # - can't prespecify this list in __init__ arguments: mutable default argument issue
            # - sklearn convention frowns upon logic that changes any parameters during __init__
            param_grid = [
                {"learning_rate": lr, "min_child_weight": mcw}
                for lr in [0.1, 0.01, 0.3]  # default is 0.3
                for mcw in [1, 5]  # default is 1
                # Other possibilities: "max_depth": [3, 6], "gamma": [0, 0.25], "colsample_bytree": [0.5, 1]
            ]

        # Choose metric and objective
        # For objective, see https://github.com/dmlc/xgboost/blob/73713de6016163252958463147c9c6cd509e79b1/python-package/xgboost/sklearn.py#L1477-L1488 and https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
        unique_classes = np.unique(y)  # Assuming y is your target variable
        if len(unique_classes) == 2:
            # binary
            metric = self.metric_binary
            extra_params = {"objective": self.objective_binary}
        else:
            # multiclass
            metric = self.metric_multiclass
            extra_params = {
                "objective": self.objective_multiclass,
                "num_class": len(unique_classes),
            }

        # Execute the cross-validation splitting with groups argument, if provided.
        # Then we will pass the computed indices to xgboost's built-in cv method.
        # The reason to do this is that xgboost's built-in cv method will not feed the groups argument through.
        # First, detect whether cv accepts groups or not:
        def accepts_groups(cv_method):
            params = inspect.signature(cv_method).parameters
            return "groups" in params

        if accepts_groups(self.cv.split):
            cv_splits = list(self.cv.split(X, y, groups=groups))
        else:
            cv_splits = list(self.cv.split(X, y))

        # Convert to DMatrix in order to use xgboost's built-in CV
        dtrain = xgb.DMatrix(data=X, label=y, weight=sample_weight)

        # Define a function to perform cross-validation for a single set of parameters
        def run_cv(params, dtrain, cv_splits):
            """
            Given one parameter dictionary, returns tuple with:
                - best_score
                - parameter dictionary again
                - best number of iterations (boosting rounds) according to early stopping
            """
            cv_results = xgb.cv(
                dtrain=dtrain,
                # using syntax compatible with python<3.9:
                params={
                    **params,
                    **extra_params,
                },
                nfold=len(cv_splits),
                folds=cv_splits,
                # Use maximum number of boosting rounds (maximum n_estimators).
                # Early stopping will find the optimal number of boosting rounds
                num_boost_round=self.max_num_trees,
                early_stopping_rounds=self.early_stopping_patience,
                metrics=metric,
                as_pandas=True,
                # This seed is ignored. (It's used to generate the folds, which has already happened by this point. Configure the seed in the cv object instead.)
                seed=0,
                # Later, consider custom callbacks: https://katerynad.github.io/XGBoost%20CV/XGBoost%20CV%20callback%20functions.htm
                # They might let us introspect the models and reject any with all-0 feature importances.
            )

            # Get best score and best number of boosting rounds (iterations) according to early stopping
            # Early stopping will terminate cv_results early. If one CV fold stops improving before others, we believe its last performance metric will be repeated for subsequent iterations of the other folds.
            average_scores_on_held_out_set = cv_results[f"test-{metric}-mean"]
            if self.metric_lower_is_better:
                # We should minimize
                best_score = np.min(average_scores_on_held_out_set)
                best_iteration = np.argmin(average_scores_on_held_out_set)
            else:
                best_score = np.max(average_scores_on_held_out_set)
                best_iteration = np.argmax(average_scores_on_held_out_set)

            # Adjusting the best iteration (Python is 0-indexed, but iteration counts start at 1)
            best_iteration += 1  # to reflect the iteration count accurately

            return best_score, params, best_iteration

        # Prepare to run in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(run_cv)(params, dtrain, cv_splits) for params in param_grid
        )

        # Find the best score and corresponding parameters
        optimization_function = min if self.metric_lower_is_better else max
        best_score, best_params, best_num_boost_round = optimization_function(
            results, key=lambda x: x[0]
        )

        # Train final model on the full dataset with the best parameters and number of boosting rounds
        # Let's use the sklearn API now.
        # Configure the classifier with the best parameters:
        self.set_params(
            # Train the final model using the best number of boosting rounds from CV, according to early stopping:
            n_estimators=best_num_boost_round,
            # Unpack all the other best parameters:
            **best_params,
        )

        # Fit the final model on the full dataset
        super().fit(X, y, sample_weight=sample_weight, **kwargs)

        return self


class XGBoostClassifierWithLabelEncodingWithCV(
    XGBoostClassifierWithLabelEncoding, XGBoostCV
):
    """
    XGBoostClassifierWithLabelEncoding but with XGBoostCV under the hood rather than XGBClassifier.
    See XGBoostClassifierWithLabelEncoding and XGBoostCV docs for details.
    """

    # We use multiple inheritance to combine the two classes.
    # Through the module resolution order, we ensure that XGBoostClassifierWithLabelEncoding.fit() will call XGBoostCV.fit() rather than XGBClassifier.fit(). XGBoostCV.fit() will of course eventually call XGBClassifier.fit().
    def __init__(
        self,
        cv: sklearn.model_selection.BaseCrossValidator,
        param_grid: Optional[dict] = None,
        max_num_trees: int = 200,
        early_stopping_patience: int = 10,
        n_jobs: int = 1,
        metric_binary: str = "logloss",
        metric_multiclass: str = "mlogloss",
        metric_lower_is_better: bool = True,
        class_weight: Optional[Union[dict, str]] = None,
        fail_if_nothing_learned: bool = True,
        **kwargs,
    ):
        super().__init__(
            cv=cv,
            param_grid=param_grid,
            max_num_trees=max_num_trees,
            early_stopping_patience=early_stopping_patience,
            n_jobs=n_jobs,
            metric_binary=metric_binary,
            metric_multiclass=metric_multiclass,
            metric_lower_is_better=metric_lower_is_better,
            class_weight=class_weight,
            fail_if_nothing_learned=fail_if_nothing_learned,
            **kwargs,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Self:
        # This looks like an unnecessary override.
        # But what it's doing is advertising that fit() supports the groups parameter here explicitly.
        # This is because XGBoostClassifierWithLabelEncodingWithCV's fit() will call XGBoostCV.fit(), which will pass groups to the CV splitter.
        # In comparison, XGBoostClassifierWithLabelEncoding's fit() will call XGBClassifier.fit(), which does not accept groups.
        return super().fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            # XGBoostClassifierWithLabelEncoding.fit() (which is super().fit() in this case) sees groups as just another kwarg to pass to its underlying fit()
            groups=groups,
            **kwargs,
        )
