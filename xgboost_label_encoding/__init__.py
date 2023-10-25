"""Xgboost Label Encoding."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.4"

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict, Optional, Union
from typing_extensions import Self
import sklearn.utils.class_weight
import xgboost as xgb


def property_available_if(check):
    # TODO: Make this a separate library
    """
    Extends sklearn.utils.metaestimators.available_if to wrap a property instead of a method.

    Usage:
    @property_available_if(...)
    @property
    def my_property(self):
        ...
    """

    # Custom version of sklearn.utils._available_if._AvailableIfDescriptor for properties.
    from sklearn.utils._available_if import _AvailableIfDescriptor

    class _AvailableIfDescriptorForProperty(_AvailableIfDescriptor):
        """Implements a conditional property using the descriptor protocol.

        Customized to wrap a property, not a method.

        Using this class to create a decorator will raise an ``AttributeError``
        if check(self) returns a falsey value. Note that if check raises an error
        this will also result in hasattr returning false.
        """

        def __get__(self, obj, owner=None):
            attr_err = AttributeError(
                f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
            )
            if obj is not None:
                if not self.check(obj):
                    raise attr_err
                # Here is our change:
                # Instead of returning MethodType(self.fn, obj),
                # just return the result of the property's getter function (self.fn(obj)).
                # This makes it behave like a regular property, but with the added conditional check.
                return self.fn(obj)
            raise attr_err

    # See https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/utils/_available_if.py#L47
    # The key difference is we use fn.fget instead of fn directly. Here's why:
    # - the function: the wrapped getter is the actual function to be executed.
    # - attribute_name: Properties don't have names, so we need to pass the name of the getter method.
    return lambda fn: _AvailableIfDescriptorForProperty(
        fn.fget, check, attribute_name=fn.fget.__name__
    )


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

        # Fit as usual
        # The fit procedure will try to access self.classes_, so our property must be temporarily set to xgboost's default classes_ value during this call to super().fit(). (See comments in the classes_ property definition below for more details.
        # fit_procedure_running_ acts as our temporarily signal to the classes_ property that we are in the middle of the fit procedure.
        self.fit_procedure_running_ = True
        super().fit(X, transformed_y, sample_weight=sample_weight, **kwargs)
        del self.fit_procedure_running_
        from packaging import version

        if version.parse(xgb.__version__) < version.parse("2"):
            # Backwards compatibility for xgboost 1.x:
            # Set self.classes_ = self.label_encoder_.classes_ after super().fit() is called.

            # The reason why is because in xgboost 1.x, classes_ is not a property, so we can set it directly.
            # Our classes_ property definition below is only for xgboost 2.x, where classes_ is indeed a property.

            # During this fit procedure, our classes_ property is not yet available. So xgboost 1.x super().fit() will set the self.classes_ attribute to a value directly.
            # This will overwrite our classes_ property definition! So our property will *never* come to life in xgboost 1.x.

            # Alternative considered: Define our classes_ property again after super().fit() has been called. Not sure how to do that.
            self.classes_ = self.label_encoder_.classes_

        return self

    # @property_available_if is a decorator that makes it so that self.classes_ is only available after fit() is called.
    # This is necessary to support both xgboost 1.x and 2.x.

    # Explanation: classes_ should only be available after fit() is called. However, we can no longer set self.classes_ = self.label_encoder_.classes_ inside fit(), because classes_ is now a property in xgboost 2.0+ and therefore can't be set.
    # Instead we overload the property and add a decorator that hides the property until fit() has been called (self.__sklearn_is_fitted__() will be True after this)

    # Also, to support a quirk in xgboost's fit method, we also make this property available *during* the fit procedure, with a different returned value.
    # That's because xgboost's fit method tries to access self.classes_ during the fit procedure and expects to find a numpy int array.
    # So we return np.arange(self.n_classes_) during the fit procedure, which is the default value that xgboost uses.

    # To recap:
    # - before the fit procedure: classes_ is not available
    # - during the fit procedure: classes_ is available, but returns np.arange(self.n_classes_)
    # - after the fit procedure: classes_ is available, and returns self.label_encoder_.classes_

    # Another important quirk: this property definition is actually never used in xgboost 1.x, because xgboost 1.x super().fit() will set the self.classes_ attribute to a value directly and therefore prevent this property definition from ever becoming available.
    @property_available_if(
        lambda self: self.__sklearn_is_fitted__()
        or hasattr(self, "fit_procedure_running_")
    )
    @property
    def classes_(self) -> np.ndarray:
        if hasattr(self, "fit_procedure_running_"):
            # We are in the middle of the fit procedure. See note above about this quirk.
            # Return the default value that xgboost uses, which is: np.arange(self.n_classes_)
            return np.arange(len(self.label_encoder_.classes_))

        return self.label_encoder_.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.label_encoder_.inverse_transform(super().predict(X))

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
