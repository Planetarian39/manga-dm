"""Velocity-field map panel generation.

Extracted from ``src-orig/figure.py``.
"""


def plot_velocity_field_panels(*args, **kwargs):
    """Generate velocity-field panel figure."""
    from src.viz.figure_panels import plot_velocity_field_panels as _impl

    return _impl(*args, **kwargs)


def plot_velocity_field_comparison(*args, **kwargs):
    """Generate velocity-field comparison figure."""
    from src.viz.figure_panels import plot_velocity_field_comparison as _impl

    return _impl(*args, **kwargs)
