"""Model factory for the ThyroidBench wrappers.

``build_model(name, **kwargs)`` dispatches to the right model constructor /
builder by name. Every branch imports its wrapper lazily so that requesting one
model does not force the heavy third-party dependencies of the others (torch,
segmentation_models_pytorch, sam2, sam3, transformers, ...) to be importable.

Supported names:
    unet, transunet, sam, sam2, medsam, medsam2, sam3, sam3_video
"""

from typing import Any

__all__ = ["build_model"]

# Canonical set of dispatchable model names (for error messages / discovery).
MODEL_NAMES = (
    "unet",
    "transunet",
    "sam",
    "sam2",
    "medsam",
    "medsam2",
    "sam3",
    "sam3_video",
)


def build_model(name: str, **kwargs: Any):
    """Construct a model / predictor by name.

    Args:
        name: One of :data:`MODEL_NAMES`.
        **kwargs: Forwarded verbatim to the underlying constructor / builder.
            Traditional baselines (``unet``, ``transunet``) accept
            ``pretrained``, ``device``, ``in_channels``. The foundation-model
            predictors accept ``device``.

    Returns:
        The constructed model (``nn.Module``) or zero-shot / video predictor
        instance, exactly as returned by the underlying wrapper.

    Raises:
        ValueError: if ``name`` is not a supported model name.
    """
    key = name.lower()

    if key == "unet":
        from .unet import build_unet
        return build_unet(**kwargs)
    if key == "transunet":
        from .transunet import build_transunet
        return build_transunet(**kwargs)
    if key == "sam":
        from .sam_wrapper import SAMPredictor
        return SAMPredictor(**kwargs)
    if key == "sam2":
        from .sam2_wrapper import SAM2Predictor
        return SAM2Predictor(**kwargs)
    if key == "medsam":
        from .medsam_wrapper import MedSAMPredictor
        return MedSAMPredictor(**kwargs)
    if key == "medsam2":
        from .medsam2_wrapper import MedSAM2Predictor
        return MedSAM2Predictor(**kwargs)
    if key == "sam3":
        from .sam3_wrapper import SAM3Predictor
        return SAM3Predictor(**kwargs)
    if key == "sam3_video":
        from .sam3_video_wrapper import build_sam3_video_predictor
        return build_sam3_video_predictor(**kwargs)

    raise ValueError(
        f"Unknown model name '{name}'. Choose from {list(MODEL_NAMES)}"
    )
