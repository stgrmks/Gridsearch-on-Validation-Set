__author__ = 'MSteger'

import numpy as np
import operator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid

class GridSearchValidationSet(BaseEstimator, TransformerMixin):

    def __init__(self, model, param_grid, X_val, y_val, scorer, postprocessing = None, verbose = 0):
        self.model = model
        self.param_grid = param_grid
        self.X_val = X_val
        self.y_val = y_val
        self.scorer = scorer
        self.postprocessing = postprocessing
        self.verbose = verbose > 0

    def fit(self, X, y, **kwargs):
        performance, gs_model_params = {}, {}

        for i, model_params in enumerate(ParameterGrid(self.param_grid)):
            self.model.set_params(**model_params)
            self.model.fit(X, y, **(kwargs or {}))
            yHat = self.model.predict_proba(self.X_val)

            if self.postprocessing is not None:
                yHat = self.postprocessing.fit_transform(yHat)
            else:
                yHat = self.model.classes_.take(np.argmax(yHat, axis=1), axis=0)

            score = self.scorer(self.y_val, yHat)
            if self.verbose: print '\nGridSearch iteration {}:\n{}: {}\nparams: {}'.format(i, self.scorer.__name__, score, model_params)
            performance[i], gs_model_params[i] = score, model_params

        sorted_performance = sorted(performance.items(), key = operator.itemgetter(1))
        best_iteration, best_score = sorted_performance[-1]
        best_params, iteration_summary = gs_model_params[best_iteration], {}
        for key,value in performance.iteritems(): iteration_summary[key] = {str(self.scorer.__name__): value, 'params': gs_model_params[key]}

        self.result_, self.best_estimator_, self.best_score_ = iteration_summary, self.model.set_params(**best_params), best_score
        return self

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y = np.random.rand(1000,10), np.random.randint(2,size=1000)
    y = np.where(y > 0, 'one', 'zero')
    X_val, y_val = X[500:], y[500:]

    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [10, 100],
        'max_depth': [2, 10]
    }

    GS = GridSearchValidationSet(model = model, param_grid = param_grid, X_val = X_val, y_val = y_val, scorer = accuracy_score, postprocessing = None, verbose = 1)
    GS.fit(X = X[:500], y = y[:500])

    print '\nResult: {}\nModel: {}\nBest Score: {}'.format(GS.result_, GS.best_estimator_, GS.best_score_)

    print 'done'