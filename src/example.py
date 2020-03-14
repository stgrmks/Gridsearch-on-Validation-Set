import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, TransformerMixin

from src.grid_search_val import GridSearchValidationSet


class sliding_majority_filter(BaseEstimator, TransformerMixin):
    def __init__(self, length=3, stride=2):
        self.length = length
        self.stride = stride

    def transform(self, X):
        yHat = self.m_classes.take(np.argmax(X, axis=1), axis=0)
        start, end = 0, self.length
        while end < yHat.shape[0]:
            yHat[start:end] = mode(yHat[start:end])[0][0]
            end += self.stride
            start += self.stride
            if end >= yHat.shape[0]:
                end = yHat.shape[0]
                yHat[start:end] = mode(yHat[start:end])[0][0]
        return yHat

    def fit(self, X, m_classes):
        self.m_classes = m_classes
        return self


if __name__ == "__main__":

    from src.utils import get_logger
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import os

    log = get_logger(os.path.basename(__file__))

    X, y = np.random.rand(1000, 10), np.random.randint(2, size=1000)
    y = np.where(y > 0, "one", "zero")
    X_val, y_val = X[500:], y[500:]

    model = RandomForestClassifier(random_state=1337)
    param_grid = {"n_estimators": [10, 100], "max_depth": [2, 10]}

    postprocessing = sliding_majority_filter(length=3, stride=2)

    GS = GridSearchValidationSet(
        model=model,
        param_grid=param_grid,
        X_val=X_val,
        y_val=y_val,
        scorer=accuracy_score,
        postprocessing=postprocessing,
        verbose=1,
    )
    GS.fit(X=X[:500], y=y[:500])

    log.info(
        "Result: {}\nModel: {}\nBest Score: {}".format(
            GS.result_, GS.best_estimator_, GS.best_score_
        )
    )
