from globals.constants import *
from sklearn.utils import all_estimators
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier


def model_factory(model_setup):
    if model_setup[NAME] == COMPOSITE:
        kwargs = model_setup[CONFIG]

        match model_setup["method"]:
            case "ada":
                base_model = kwargs.get("model", "knn")
                ensemble_params = kwargs.get(PARAMS, {})
                base_params = kwargs.get("model_params", {})

                return AdaBoostClassifier(base_estimator=catalog[base_model](**base_params), **ensemble_params)
            case "voting":
                base_models = kwargs.get("models", [
                    {NAME: "KNeighborsClassifier", PARAMS: {}},
                    {NAME: "SVC", PARAMS: {}}
                ])
                ensemble_params = kwargs.get(PARAMS, {})
                estimators = [(model[NAME], catalog[model[NAME]](**model.get(PARAMS, {}))) for model in base_models]

                return VotingClassifier(estimators=estimators, **ensemble_params)
    else:
        return catalog[model_setup[NAME]](**model_setup.get(PARAMS, {}))


catalog = { e[0]: e[1] for e in all_estimators(type_filter='classifier') }
