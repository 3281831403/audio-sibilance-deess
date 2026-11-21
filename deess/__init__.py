"""Sibilance de-esser package."""

from .deesser import (
    DeessError,
    apply_band_gain,
    band_level,
    band_split,
    deess_batch,
    detect_sibilance,
    gain_reduction,
    smooth_gain,
    validate_params,
)

__all__ = [
    "DeessError",
    "apply_band_gain",
    "band_level",
    "band_split",
    "deess_batch",
    "detect_sibilance",
    "gain_reduction",
    "smooth_gain",
    "validate_params",
]
