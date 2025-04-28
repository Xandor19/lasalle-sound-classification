from base import BaseCustom
from globals.constants import *
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier


oversampling_catalog = {
    "smote": SMOTE,
    "adasyn": ADASYN
}


class ModelWrapper(BaseCustom, ClassifierMixin):
    """
    Wraps the usage of a classifier to allow non-pipeline compatible operations like oversampling

    ## Params:
    - model_setup: Dictionary with the different configurations available for the underlying models, including
                   class balancing ("class-balance" set to one of the available techniques) strategy and parameters ("balancer-params" 
                   with nested dict), classification method (ada for AdaBoost, voting for VotingClassifier or any 
                   class name of the Sci-kit Learn available classifiers). For ensemble nested dictionaries must
                   be provided indicating base models and parameters
    """
    def __init__(self, name="KNeighborsClassifier", params={}, class_balance=None, balancer_params=None, config={}):
        self.name=name
        self.params = params
        self.class_balance = class_balance
        self.balancer_params = balancer_params
        self.config = config

        if name == COMPOSITE:
            match config["method"]:
                case "ada":
                    # Ada Boost based ensemble, get the base model and prepare the parameters
                    base_model = config.get("model", "KNeighborsClassifier")
                    base_params = config.get("model_params", {})
                    self.estimator = AdaBoostClassifier(base_estimator=catalog[base_model](**base_params), **params)
                    return 
                case "voting":
                    # Voting ensemble, get all models and corresponding setup 
                    base_models = config.get("models", [
                        {NAME: "KNeighborsClassifier", PARAMS: {}},
                        {NAME: "SVC", PARAMS: {}}
                    ])
                    ensemble_params = config.get(PARAMS, {})
                    estimators = [(model[NAME], catalog[model[NAME]](**model.get(PARAMS, {}))) for model in base_models]

                    self.estimator = VotingClassifier(estimators=estimators, **ensemble_params)
        else:
            # Use default Sci-kit Learn models
            self.estimator = catalog[name](**params)

    def fit(self, X, y=None):
        balancing = oversampling_catalog.get(self.class_balance)

        if balancing:
            # Apply defined balancer to the data
            X, y = balancing(**self.balancer_params).fit_resample(X, y)

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
