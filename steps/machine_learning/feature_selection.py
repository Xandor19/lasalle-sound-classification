import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class KBestSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on generalized rankings to select the best features

    ## Params:
    - method: Method to use for feature selection. Can be an string specifying the method or directly a 
              callable compatible with Sklearn's SelectKBest. Currently available methods are: "mutual_info" (direct
              map to sklearn's mutual_info_classif) and "fisher_score" to compute inter-intra class distance.
    - k: number of features to keep.
    """
    def __init__(self, method='fisher_score', k=20, **kwargs):
        self.method = method
        self.k = k
        self.methods = {
            "mutual_info": mutual_info_classif,
            "fisher_score": self.fisher_score
        }
        if method not in self.methods:
            raise ValueError(f"Invalid method: {method}. Available methods are: {list(self.methods.keys())}")
        self.selector = None
        self.selected_columns_ = None

    def fit(self, X, y):
        nas = X.isna().sum()
        print(nas[nas > 0])
        self.column_names_ = X.columns
        self.selector = SelectKBest(score_func=self.method if callable(self.method) else self.methods[self.method] , k=self.k)
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        self.selected_columns_ = list(X.columns[mask])

        return self

    def transform(self, X):
        return X[self.selected_columns_]

    def fisher_score(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.values  # ensure numpy array

        classes = np.unique(y)
        scores = []

        for i in range(X.shape[1]):
            inter_class = np.var([X[y == c, i].mean() for c in classes])
            intra_class = np.mean([X[y == c, i].var() for c in classes])
            score = inter_class / (intra_class + 1e-6)
            scores.append(score)

        return np.array(scores)


class ModelBasedSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on a model's results with the features

    ## Params:
    - method: Method to use for feature selection. Can be "rfe" for recursive feature elimination or "rf_importance" for
              random forest feature importance. If "rfe", a model instance must be provided in the kwargs with name "model".
              The same model type that would be used on inference should be provided
    - k: number of features to keep.
    - kwargs: 
    """
    def __init__(self, method='rfe', k=20, model=None, **kwargs):
        self.method = method
        self.k = k
        self.model = model
        self.selector = None
        self.selected_columns_ = None

    def fit(self, X, y):
        if self.method == 'rfe':
            if not self.model:
                raise ValueError("Model must be provided for RFE method.")
            
            self.selector = RFE(self.model, n_features_to_select=self.k)
            self.selector.fit(X, y)
            mask = self.selector.get_support()
            self.selected_columns_ = list(X.columns[mask])

        elif self.method == 'rf_importance':
            model = RandomForestClassifier()
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:self.k]
            self.selected_columns_ = [self.column_names_[i] for i in indices]

        return self

    def transform(self, X):
        return X[self.selected_columns_]


catalog = {
    "kbest": KBestSelector,
    "modelbased": ModelBasedSelector
}