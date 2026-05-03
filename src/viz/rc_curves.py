"""Rotation-curve plot utilities.

Extracted from ``src-orig/figure.py`` and ``src-orig/main.py``.
"""


def plot_rc_fit_summary_comparison(*args, **kwargs):
    """Generate RC fit summary comparison plot (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import figure as _fig
    return _fig.plot_rc_fit_summary_comparison(*args, **kwargs)


def plot_rc_fit_summary_panels(*args, **kwargs):
    """Generate RC fit summary panel plot (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import figure as _fig
    return _fig.plot_rc_fit_summary_panels(*args, **kwargs)
