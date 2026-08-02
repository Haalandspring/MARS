"""MARS: Morphology-Aware RFI Segmentation.

Model symbols are loaded lazily so lightweight metadata/augmentation tooling can
be inspected before the optional heavyweight runtime is initialized.
"""

__all__ = [
    "PAPER_PARAMETER_COUNT",
    "TRTShapeUNet512",
    "build_model",
    "build_paper_model",
    "count_parameters",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import model

    return getattr(model, name)
