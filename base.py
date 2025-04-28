from sklearn.base import BaseEstimator, TransformerMixin


class BaseCustom(BaseEstimator):
    """
    Template class for all custom components that handles the update of parameters and the application
    of batch transformations
    """
    def transform(self, X):
        return [self._apply_transform(x) for x in X]
    

    def set_params(self, **params):
        """Override to avoid failure when receiving more parameters than expected, simply ignoring them"""
        for p, v in params.items():
            try:
                super().set_params(**{p: v})
            except:
                pass

        return self
    
    def _apply_transform(self, x):
        """Base method for individual instances transforming"""
        raise NotImplementedError("Subclasses should implement this method.")

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