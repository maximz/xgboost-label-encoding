"""Xgboost Label Encoding."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.4"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict, Optional, Union
from typing_extensions import Self
import sklearn.utils.class_weight
import xgboost as xgb


class XGBoostClassifierWithLabelEncoding(xgb.XGBClassifier):
    """
    Wrapper around XGBoost XGBClassifier with label encoding for the target y label.

    Native XGBoost doesn't support string labels, and XGBClassifier's `use_label_encoder` property was removed in 1.6.0.
    Unfortunately, sklearn's `LabelEncoder` for `y` target values does not play well with sklearn pipelines.

    Our workaround: wrap XGBClassifier in this wrapper for automatic label encoding of y.
    Use this in place of XGBClassifier, and `y` will automatically be label encoded.

    Additional features:
    - automatic class weight rebalancing as in sklearn
    - automatic renaming of column names passed through to xgboost to avoid xgboost error: "feature_names must be string, and may not contain [, ] or <"
    """

    _original_feature_names_: Optional[np.ndarray]
    _transformed_feature_names_: Optional[np.ndarray]

    def __init__(
        self, class_weight: Optional[Union[dict, str]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.class_weight = class_weight

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

        # Drop "class_weight" from params
        if "class_weight" in params:  # it should be
            del params["class_weight"]

        return params
