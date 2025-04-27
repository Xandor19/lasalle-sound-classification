from globals.constants import *
from steps.step_factory import step_factory
from sklearn.pipeline import Pipeline
from steps.machine_learning.models import model_factory as ml_factory
from steps.machine_learning.feature_extraction import FeatureExtractor


def multi_model_ranking(config):
    experiments = {}

    for experiment_config in config['experiments']:
        model_pipeline = ml_pipeline(config) if config['model-type'] == 'ml' else dl_pipeline(config)
        
        experiments[config['experiment-name']] = PIPELINE(model_pipeline, experiment_config)


def PIPELINE(model_pipeline, config):
    fixer = step_factory(FIXER, config.get(FIXER))
    detector_params = config.get(FIXER, {}).get(PARAMS, {})

    return Pipeline(steps=[
        (DETECTOR, step_factory(DETECTOR, config.get(DETECTOR, {}).get(NAME), instantiate=False)(fixer=fixer, **detector_params)),
        (DENOISER, step_factory(DENOISER, config.get(DENOISER))),
        (NORMALIZER, step_factory(NORMALIZER, config.get(NORMALIZER))),
        (MODEL, model_pipeline)
    ])


def dl_pipeline(config):
    pass


def ml_pipeline(config):
    model = ml_factory(config[MODEL])
    selector_params = config.get(FEATURE_SELECTOR, {}).get(PARAMS, {})

    return Pipeline(steps=[
        (FEATURE_EXTRACTOR, FeatureExtractor(config.get(FEATURE_EXTRACTOR, None))),
        (FEATURE_SELECTOR, step_factory(FEATURE_SELECTOR, config.get(FEATURE_SELECTOR, {}).get(NAME), instantiate=False)(model=model, **selector_params)),
        (MODEL, model)
    ])
