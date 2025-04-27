from globals.constants import *
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import all_estimators
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier


oversampling_catalog = {
    "smote": SMOTE,
    "adasyn": ADASYN
}


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps the usage of a classifier to allow non-pipeline compatible operations like oversampling

    ## Params:
    - model_setup: Dictionary with the different configurations available for the underlying models, including
                   class balancing ("class-balance" set to one of the available techniques) strategy and parameters ("balancer-params" 
                   with nested dict), classification method (ada for AdaBoost, voting for VotingClassifier or any 
                   class name of the Sci-kit Learn available classifiers). For ensemble nested dictionaries must
                   be provided indicating base models and parameters
    """
    def __init__(self, model_setup):
        self.imbalance_handling = model_setup.get("class-balance", "ignore")
        self.balancing_setup = model_setup.get("balancer-params", {})

        if model_setup[NAME] == COMPOSITE:
            kwargs = model_setup[CONFIG]

            match model_setup["method"]:
                case "ada":
                    base_model = kwargs.get("model", "knn")
                    ensemble_params = kwargs.get(PARAMS, {})
                    base_params = kwargs.get("model_params", {})
                    self.estimator = AdaBoostClassifier(base_estimator=catalog[base_model](**base_params), **ensemble_params)
                    return 
                case "voting":
                    base_models = kwargs.get("models", [
                        {NAME: "KNeighborsClassifier", PARAMS: {}},
                        {NAME: "SVC", PARAMS: {}}
                    ])
                    ensemble_params = kwargs.get(PARAMS, {})
                    estimators = [(model[NAME], catalog[model[NAME]](**model.get(PARAMS, {}))) for model in base_models]

                    self.estimator = VotingClassifier(estimators=estimators, **ensemble_params)
        else:
            self.estimator = catalog[model_setup[NAME]](**model_setup.get(PARAMS, {}))

    def fit(self, X, y=None):
        balancing = oversampling_catalog.get(self.imbalance_handling)

        if balancing:
            # Apply defined balancer to the data
            X, y = balancing(**self.balancing_setup).fit_resample(X, y)

        self.estimator.fit(X, y)

        return self

    # Delegate the call to the estimator

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight = None):
        return self.estimator.score(X, y, sample_weight)
    
    def set_params(self, **params):
        self.estimator.set_params(**params)
        return super().set_params(**params)


catalog = { e[0]: e[1] for e in all_estimators(type_filter='classifier') }
