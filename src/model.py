"""Model construction."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    Bag trees over independent random sparse linear combinations of the features.
    Tune tree parameters through `model__estimator__tree__<param>` and projection
    parameters through `model__estimator__project__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


def adaboost(max_depth: int = 1, **params):
    """Boost shallow trees; tune their depth via `model__estimator__max_depth`."""
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


_MODELS = {
    "dummy": DummyClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


class NoFeaturesSelectedError(ValueError):
    """The requested vote threshold retained no features on these training rows."""


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep features chosen by at least min_votes of RF, boosting and L1 logistic regression."""

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        if self.min_votes not in (1, 2, 3):
            raise ValueError("min_votes must be 1, 2 or 3")
        X, y = check_X_y(X, y, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                        random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(penalty="l1", C=0.01, solver="liblinear",
                                    random_state=self.random_state).fit(X, y)
        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean())
            + (np.abs(lasso.coef_[0]) > 1e-8)
        )
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            raise NoFeaturesSelectedError("VoteSelector selected no features; lower min_votes")
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def build_pipeline(preprocessing_config: dict, model_config: dict,
                   selection_config: dict = None, memory=None) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing and model as one
    Pipeline, so every learned step is re-fit on the training part of each fold. Step names
    address hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
    Selection is learned inside each fold. memory caches preprocessing and selection
    for repeated fits on the same training rows and settings during tuning.
    """
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", VoteSelector(
            min_votes=selection_config.get("min_votes", 2),
            random_state=selection_config.get("random_state"),
        )))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)
