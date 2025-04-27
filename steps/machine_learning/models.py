from globals.constants import *
from sklearn.utils import all_estimators
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier


def model_factory(model_setup):
    match model_setup[NAME]:
        case "ada":
            kwargs = model_setup["config"]
            base_model = kwargs.get("model", "knn")
            ensemble_params = kwargs.get(PARAMS, {})
            base_params = kwargs.get("model_params", {})

            return AdaBoostClassifier(base_estimator=catalog[base_model](**base_params), **ensemble_params)
        case "voting":
            kwargs = model_setup["config"]
            base_models = kwargs.get("models", [
                {NAME: "knn", PARAMS: {}},
                {NAME: "svm", PARAMS: {}}
            ])
            ensemble_params = kwargs.get(PARAMS, {})
            estimators = [(model[NAME], catalog[model[NAME]](**model.get(PARAMS, {}))) for model in base_models]

            return VotingClassifier(estimators=estimators, **ensemble_params)
        case _:
            return catalog[model_setup[NAME]](**model_setup.get(PARAMS, {}))


catalog = { e[0]: e[1] for e in all_estimators(type_filter='classifier') }
