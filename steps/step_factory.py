import steps.detector as detectors
import steps.outlier_fixer as fixers
import steps.denoiser as denoisers
import steps.normalizer as normalizers
import steps.machine_learning.feature_selector as selectors
import steps.machine_learning.feature_tuner as tuners
from globals.constants import *
from base import EntityTransformer


catalog = {
    DETECTOR: detectors.catalog,
    FIXER: fixers.catalog,
    DENOISER: denoisers.catalog,
    NORMALIZER: normalizers.catalog,
    FEATURE_SELECTOR: selectors.catalog,
    FEATURE_TUNING: tuners.catalog
}


def step_factory(step, method=None, instantiate=True):
    """
    Factory method to generalize the creation of pipeline steps based on each step available catalog

    ## Params:
    - step: Step type to create. Can be one of the defined step constants.
    - method: Name of the method to use for the step or a dictionary of "name" and optionally "params" for in-place instantiation. 
              If None, a default non-op step is created
    - instantiate: If True, the method parameter should be dictionary having at least "name" key for the method name. If False, the class is returned
    
    ## Returns:
    - An instance of the specified step or a callable class if instantiate is False.
    """
    if method:
        if instantiate:
            params = method[PARAMS] if PARAMS in method.keys() else {}
            return catalog[step][method[NAME]](**params)
        else:
            return catalog[step][method]
    else:
        if instantiate:
            return EntityTransformer()
        else:
            return EntityTransformer
