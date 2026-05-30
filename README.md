# xgboost-label-encoding

`xgboost-label-encoding` provides small sklearn-style wrappers around
`xgboost.XGBClassifier` for classification workflows where the target labels are
strings or other non-numeric values.

XGBoost trains on numeric class labels. This package encodes `y` during `fit`,
trains the underlying XGBoost classifier, and decodes predictions back to the
original labels. It is intended to be used as a drop-in estimator in places where
manually applying `sklearn.preprocessing.LabelEncoder` to the target would be
awkward.

## Installation

```bash
pip install xgboost_label_encoding
```

The package requires Python 3.8+ and installs against `xgboost<2`.

For local development:

```bash
pip install -r requirements_dev.txt
pip install -e .
make test
```

## Usage

Use `XGBoostClassifierWithLabelEncoding` in place of `xgboost.XGBClassifier`:

```python
from xgboost_label_encoding import XGBoostClassifierWithLabelEncoding

clf = XGBoostClassifierWithLabelEncoding(
    n_estimators=100,
    class_weight="balanced",
)

clf.fit(X_train, y_train)  # y_train may contain labels like "Healthy" or "HIV"

labels = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
classes = clf.classes_
```

Most XGBoost classifier parameters are passed through unchanged. The wrapper adds
these project-specific options:

- `class_weight`: passed to `sklearn.utils.class_weight.compute_sample_weight`;
  if `sample_weight` is also supplied, the two weights are multiplied.
- `fail_if_nothing_learned`: defaults to `True`; raises `ValueError` after
  fitting if all feature importances are zero.

## Cross-Validated Fitting

`XGBoostClassifierWithLabelEncodingWithCV` combines label encoding with
cross-validation over XGBoost parameters:

```python
from sklearn.model_selection import StratifiedKFold
from xgboost_label_encoding import XGBoostClassifierWithLabelEncodingWithCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = XGBoostClassifierWithLabelEncodingWithCV(
    cv=cv,
    max_num_trees=200,
    early_stopping_patience=10,
    class_weight="balanced",
)

clf.fit(X_train, y_train)
```

During `fit`, the CV wrapper:

- builds a small default grid of `learning_rate` and `min_child_weight` values
  unless `param_grid` is provided;
- runs `xgboost.cv` with early stopping for each parameter set;
- selects the best parameter set and number of boosting rounds;
- fits the final classifier on the full training data.

If the provided CV splitter accepts a `groups` argument, `groups` can be passed
to `fit`.

## Behavior And Limitations

- Training data must contain at least two classes.
- `predict` returns original labels, not encoded integers.
- `predict_proba` returns one probability column per class in `clf.classes_`.
- For pandas DataFrame inputs, feature names containing `[`, `]`, or `<` are
  renamed internally before reaching XGBoost. `feature_names_in_` still exposes
  the original feature names, and the same renaming is applied during
  `predict` and `predict_proba`.
- `XGBoostCV` is also available as a standalone helper for numeric-label
  XGBoost classification with CV-selected hyperparameters and tree count.

## Development

Useful local commands:

```bash
make test
make lint
make docs
make dist
```
