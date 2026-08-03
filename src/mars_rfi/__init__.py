"""MARS: Morphology-Aware RFI Segmentation and mitigation.

Core model and mitigation symbols are loaded lazily so importing package
metadata does not initialize the optional filterbank/CUDA runtime.
Operational search helpers live in :mod:`mars_rfi.search`.
"""

_MODEL_EXPORTS = {
    "PAPER_PARAMETER_COUNT",
    "TRTShapeUNet512",
    "build_model",
    "build_paper_model",
    "count_parameters",
}
_MITIGATION_EXPORTS = {"load_pipeline_config", "mitigate_filterbank"}

__all__ = sorted(_MODEL_EXPORTS | _MITIGATION_EXPORTS)

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import model

        return getattr(model, name)
    if name in _MITIGATION_EXPORTS:
        from . import mitigate

        return getattr(mitigate, name)
    raise AttributeError(name)
