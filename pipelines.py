from globals.constants import *
from steps.input import tagged_audios
from steps.step_factory import step_factory
from steps.machine_learning.feature_extraction import FeatureExtractor
from sklearn.pipeline import Pipeline


def multi_model_ranking(config):
    experiments = {}

    for experiment_config in config['experiments']:
        model_pipeline = ml_pipeline(config) if config['model-type'] == 'ml' else dl_pipeline(config)
        
        experiments[config['experiment-name']] = run_pipeline(model_pipeline, experiment_config)


def run_pipeline(model_pipeline, config):
    X, y = tagged_audios(config)
    fixer = step_factory(FIXER, with_fallback(config, FIXER))
    detector_params = with_fallback(config, FIXER, PARAMS, False)

    pipeline = Pipeline(steps=[
        (DETECTOR, step_factory(DETECTOR, with_fallback(config, DETECTOR, NAME), instantiate=False)(fixer=fixer, **detector_params)),
        (DENOISER, step_factory(DENOISER, with_fallback(config, DENOISER))),
        (NORMALIZER, step_factory(NORMALIZER, with_fallback(config, NORMALIZER))),
        (MODEL, model_pipeline)
    ])

    pipeline.fit(X, y)
    pipeline.predict(X)


def dl_pipeline(config):
    pass


def ml_pipeline(config):
    model = step_factory(MODEL, config[MODEL])
    selector_params = with_fallback(config, FEATURE_SELECTOR, PARAMS, False)

    return Pipeline(steps=[
        (FEATURE_EXTRACTOR, FeatureExtractor(with_fallback(config, FEATURE_EXTRACTOR))),
        (FEATURE_SELECTOR, step_factory(FEATURE_SELECTOR, with_fallback(config, FEATURE_SELECTOR, NAME), instantiate=False)(model=model, **selector_params)),
        (MODEL, model)
    ])


def with_fallback(conf, key, inner=None, void_return=True):
    if inner:
        result = conf[key][inner] if key in conf.keys() and inner in conf[key].keys() else {}
    else:
        result = conf[key] if key in conf.keys() else {}

    if not result and void_return:
        return None
    else:
        return result
