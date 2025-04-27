import numpy as np
from globals.constants import *
from steps.step_factory import step_factory
from sklearn.pipeline import Pipeline
from steps.machine_learning.models import model_factory as ml_factory
from steps.machine_learning.feature_extraction import FeatureExtractor
from sklearn.model_selection import RandomizedSearchCV, cross_validate


def PIPELINE(model_pipeline, config):
    fixer = step_factory(FIXER, config.get(FIXER))
    detector_params = config.get(FIXER, {}).get(PARAMS, {})

    return Pipeline(steps=[
            (DETECTOR, step_factory(DETECTOR, config.get(DETECTOR, {}).get(NAME), instantiate=False)(fixer=fixer, **detector_params)),
            (DENOISER, step_factory(DENOISER, config.get(DENOISER))),
            (NORMALIZER, step_factory(NORMALIZER, config.get(NORMALIZER)))
        ] + model_pipeline
    )


def dl_pipeline(config):
    pass


def ml_pipeline(config):
    model = ml_factory(config[MODEL])
    selector_params = config.get(FEATURE_SELECTOR, {}).get(PARAMS, {})

    return [
        (FEATURE_EXTRACTOR, FeatureExtractor(config.get(FEATURE_EXTRACTOR, None))),
        (FEATURE_SELECTOR, step_factory(FEATURE_SELECTOR, config.get(FEATURE_SELECTOR, {}).get(NAME), instantiate=False)(model=model, **selector_params)),
        (MODEL, model)
    ]


def pipeline_factory(config):
    model_pipeline = ml_pipeline(config["steps"]) if config['model-type'] == 'ml' else dl_pipeline(config["steps"])

    return PIPELINE(model_pipeline, config["steps"])


def param_optimization(X, y, config):
    metrics = config.get(METRICS, ["f1_weighted"])
    steps = config["steps"]
    tuning_dict = {}

    for step, step_conf in steps.items():
        if NAME in step_conf.keys() and step_conf[NAME] == COMPOSITE:
            inners = step_conf[CONFIG] if isinstance(step_conf[CONFIG], list) else [step_conf[CONFIG]]

            for inner in inners:
                for param, param_conf in inner.get("optimize-params", {}).items():
                    # Registry the params search configuration of the nested component
                    tuning_dict[f"{step}__{inner[NAME]}__{param}"] = param_conf
        else:
            for param, param_conf in step_conf.get("optimize-params", {}).items():
                # Registry the params search configuration
                tuning_dict[f"{step}__{param}"] = param_conf

    print(f"tuning_dict: {tuning_dict}")

    pipeline = pipeline_factory(config)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=tuning_dict,
        scoring=metrics,
        n_iter=config.get("search-iters", 50),
        n_jobs=config.get("search-parallel", -1),
        cv=config.get("number-of-folds", 5),
        refit=config.get("search-refit", metrics[0]),
        verbose=3
    )
    return search.fit(X, y)


def multi_model_ranking(X, y, config):
    experiments = {}

    for experiment_config in config['experiments']:
        name = experiment_config['name']
        mode = experiment_config.get('mode', 'cross-val')

        match mode:
            case 'search':
                searcher = param_optimization(X, y, experiment_config)
                estimator = searcher.best_estimator_
                metrics =  searcher.cv_results_.items()

            case 'cross-val':
                estimator = pipeline_factory(config)
                metrics = cross_validate(estimator)

        experiments[name] = { m.replace("test_", ""): np.mean(r) for m, r in metrics }

    return experiments