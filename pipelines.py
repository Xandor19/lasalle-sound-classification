import json
import numpy as np
from globals.constants import *
from steps.step_factory import step_factory
from sklearn.pipeline import Pipeline
from steps.machine_learning.models import ModelWrapper
from steps.machine_learning.feature_extractor import FeatureExtractor
from steps.deep_learning.spectrogram_extractor import SpectrogramExtractor
from steps.deep_learning.models import DLModelWrapper
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate


def PIPELINE(model_pipeline, config):
    """
    Defines the base preprocessing pipeline and appends the specific model pipeline to it

    ** Params:
    - model_pipeline: List of sklearn pipeline-ready steps for feature extraction and model training
    - config: Dictionary with the configuration for each step of the pipeline
    """
    fixer = step_factory(FIXER, config.get(FIXER))
    detector_params = config.get(DETECTOR, {}).get(PARAMS, {})

    return Pipeline(steps=[
            (DETECTOR, step_factory(DETECTOR, config.get(DETECTOR, {}).get(NAME), instantiate=False)(fixer=fixer, **detector_params)),
            (DENOISER, step_factory(DENOISER, config.get(DENOISER))),
            (NORMALIZER, step_factory(NORMALIZER, config.get(NORMALIZER)))
        ] + model_pipeline
    )


def ml_pipeline(config):
    """
    Builds the pipeline for Machine Learning tasks

    ** Params
    - config: Dictionary with the configuration for each step of the pipeline
    """
    model = ModelWrapper(**config[MODEL])
    selector_params = config.get(FEATURE_SELECTOR, {}).get(PARAMS, {})

    return [
        (FEATURE_EXTRACTOR, FeatureExtractor(config.get(FEATURE_EXTRACTOR, None))),
        (FEATURE_SELECTOR, step_factory(FEATURE_SELECTOR, config.get(FEATURE_SELECTOR, {}).get(NAME), instantiate=False)(model=model.estimator, **selector_params)),
        (FEATURE_TUNING, step_factory(FEATURE_TUNING, config.get(FEATURE_TUNING))),
        (MODEL, model)
    ]


def dl_pipeline(config):
    """
    Builds the pipeline for Deep Learning tasks

    ** Params
    - config: Dictionary with the configuration for each step of the pipeline
    """
    return [
        (FEATURE_EXTRACTOR, SpectrogramExtractor(**config.get(FEATURE_EXTRACTOR, {}).get(PARAMS, {}))),
        (MODEL, DLModelWrapper(**config.get(MODEL, {}).get(PARAMS, {})))
    ]


def pipeline_factory(config):
    """
    Generates an end-to-end pipeline with the specified configuration

    * Params
    - config: Dictionary with the configuration for each step of the pipeline and the kind of learning to use
    """
    model_pipeline = ml_pipeline(config["steps"]) if config['model-type'] == 'ml' else dl_pipeline(config["steps"])

    return PIPELINE(model_pipeline, config["steps"])


def param_optimization(X, y, config):
    """
    Performs a randomized search with cross validation using the specified data. It is an end-to-end
    function that handles pipelines creation from configuration and metrics preparation

    ** Params
    - X: iterable with audio waveforms
    - y: classes of each audio
    - config: Dictionary with the full process config, including both pipeline components and validation parameters
    """
    metrics = config.get(METRICS, ["f1_weighted"])
    refit_target = config.get("search-refit", metrics[0] if isinstance(metrics, list) else next(iter(metrics)))
    steps = config["steps"]
    tuning_dict = {}
    to_remove = set()

    # Collect parameters to optimize
    for step, step_conf in steps.items():
        if NAME in step_conf.keys() and step_conf[NAME] == COMPOSITE:
            inners = step_conf[CONFIG] if isinstance(step_conf[CONFIG], list) else [step_conf[CONFIG]]

            for i, inner in enumerate(inners):
                for param, param_conf in inner.get("optimize-params", {}).items():
                    # Registry the params search configuration of the nested component
                    tuning_dict[f"{step}__{inner[NAME]}__{param}"] = param_conf
                    to_remove.add(".".join([step, CONFIG, i]))
        else:
            for param, param_conf in step_conf.get("optimize-params", {}).items():
                # Registry the params search configuration
                tuning_dict[f"{step}__{param}"] = param_conf
                to_remove.add(step)

    for remove in to_remove:
        # Remove all "optimize-params" keys
        access = remove.split(".")
        tmp = steps

        for inner in access:
            tmp = tmp[inner]

        del tmp["optimize-params"]

    pipeline = pipeline_factory(config)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=tuning_dict,
        scoring=metrics,
        n_iter=config.get("search-iters", 50),
        n_jobs=config.get("search-parallel", -1),
        cv=StratifiedKFold(config.get("number-of-folds", 5)),
        refit=refit_target,
        verbose=3
    )
    return search.fit(X, y)


def multi_experiment_runner(X, y, config, save_on_experiment=True):
    """
    Handles the run of multiple experiments, defined as a set components to form a pipeline
    and an specific configuration or parameters range

    ** Params
    - X: iterable with audio waveforms
    - y: classes of each audio
    - config: Dictionary with the full config for each experiment 
    """
    experiments = {}

    for experiment_config in config["experiments"]:
        name = experiment_config[NAME]
        mode = experiment_config.get("mode", "cross-val")

        match mode:
            case "search":
                searcher = param_optimization(X, y, experiment_config)
                estimator = searcher.best_estimator_
                metrics = searcher.cv_results_
                params = searcher.best_params_

            case "cross-val":
                estimator = pipeline_factory(experiment_config)
                metrics = cross_validate(
                    estimator, 
                    X,
                    y,
                    cv=StratifiedKFold(experiment_config.get("number-of-folds", 5)), 
                    n_jobs=experiment_config.get("search-parallel", -1),
                    scoring=experiment_config.get(METRICS, ["f1_weighted"]))
                params = None
                
        # Extract metrics from the experiment and save best parameters if a optimization process was performed
        experiments[name] = { METRICS: { m.replace("test_", ""): np.mean(r) for m, r in metrics.items() if isinstance(r, np.ndarray) and not "param" in m }}

        if params:
            experiments[name][PARAMS] = params

        if save_on_experiment:
            with open(f"output/{name}_tmp.json", "w") as tmp:
                tmp.write(json.dumps(experiments[name]))

    return experiments
