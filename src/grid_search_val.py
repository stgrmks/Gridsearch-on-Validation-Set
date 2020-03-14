import os
import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
    RegressorMixin,
    ClassifierMixin,
)
from typing import Union, Dict, Callable
from sklearn.model_selection import ParameterGrid
from src.utils import get_logger

log = get_logger(os.path.basename(__file__))


class GridSearchValidationSet(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model: Union[ClassifierMixin, RegressorMixin],
        param_grid: Dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scorer: Callable,
        postprocessing: Union[BaseEstimator, TransformerMixin] = None,
        verbose: int = 0,
    ):
        self.model: Union[ClassifierMixin, RegressorMixin] = model
        self.param_grid: Dict = param_grid
        self.X_val: np.ndarray = X_val
        self.y_val: np.ndarray = y_val
        self.scorer: Callable = scorer
        self.postprocessing: Union[BaseEstimator, TransformerMixin] = postprocessing
        self.verbose: int = verbose > 0
        self.result_: Dict = None
        self.best_estimator_: Union[ClassifierMixin, RegressorMixin] = None
        self.best_score_: float = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        search_summary: Dict = dict()
        best_iteration: int = 0
        best_score: float = 0.0

        for i, model_params in enumerate(ParameterGrid(self.param_grid)):
            model: Union[ClassifierMixin, RegressorMixin] = clone(self.model)
            model.set_params(**model_params)
            model.fit(X, y, **(kwargs or {}))

            if self.postprocessing is not None:
                yHat: np.ndarray = model.predict_proba(self.X_val) if isinstance(
                    model, ClassifierMixin
                ) else model.predict(self.X_val)
                yHat = self.postprocessing.fit_transform(
                    X=yHat, m_classes=model.classes_
                )
            else:
                yHat: np.ndarray = model.predict(self.X_val)

            score: float = self.scorer(self.y_val, yHat)
            if score > best_score:
                best_iteration = i
                best_score = score
            if self.verbose:
                log.info(
                    "GridSearch iteration {}: {}: {} params: {}".format(
                        i, self.scorer.__name__, score, model_params
                    )
                )

            search_summary[i] = {self.scorer.__name__: score, "params": model.get_params(), "model": model}
        self.result_ = search_summary
        self.best_estimator_ = search_summary[best_iteration]["model"]
        self.best_score_ = best_score

        return self


if __name__ == "__main__":
    print("done")
