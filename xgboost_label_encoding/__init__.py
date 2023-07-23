"""Xgboost Label Encoding."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.2"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union
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
    """

    def __init__(
        self, class_weight: Optional[Union[dict, str]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def fit(
        self,
        X: np.ndarray,
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

        # fit as usual
        super().fit(X, transformed_y, sample_weight=sample_weight, **kwargs)

        # set classes_
        self.classes_: np.ndarray = self.label_encoder_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.label_encoder_.inverse_transform(super().predict(X))
