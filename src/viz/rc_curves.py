"""Rotation-curve plot utilities.

Extracted from ``src-orig/figure.py`` and ``src-orig/main.py``.
"""


def plot_rc_fit_summary_comparison(*args, **kwargs):
    """Generate RC fit summary comparison plot."""
    from src.viz.figure_panels import plot_rc_fit_summary_comparison as _impl

    return _impl(*args, **kwargs)


def plot_rc_fit_summary_panels(*args, **kwargs):
    """Generate RC fit summary panel plot."""
    from src.viz.figure_panels import plot_rc_fit_summary_panels as _impl

    return _impl(*args, **kwargs)
