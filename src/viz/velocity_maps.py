"""Velocity-field map panel generation.

Extracted from ``src-orig/figure.py``.
"""


def plot_velocity_field_panels(*args, **kwargs):
    """Generate velocity-field panel figure (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import figure as _fig
    return _fig.plot_velocity_field_panels(*args, **kwargs)


def plot_velocity_field_comparison(*args, **kwargs):
    """Generate velocity-field comparison figure (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import figure as _fig
    return _fig.plot_velocity_field_comparison(*args, **kwargs)
