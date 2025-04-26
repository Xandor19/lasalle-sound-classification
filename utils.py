from sklearn.base import BaseEstimator, TransformerMixin


class EntityTransformer(BaseEstimator, TransformerMixin):
    """
    A fallback class to skip one pipeline step.
    """

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X