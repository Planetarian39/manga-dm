"""Top-level paper-figure generation (multi-panel composites).

Extracted from ``src-orig/figure.py`` and ``src-orig/m200.py``.
"""

from dataclasses import dataclass


@dataclass
class GalaxyFigureData:
    """Data container for a single galaxy's figure preparation.

    Migrated from ``src-orig/figure.py`` (placeholder).
    """
    plate_ifu: str
    rc_params: dict | None = None
    nfw_params: dict | None = None


def plot_m200_c_relation_all(*args, **kwargs):
    """Generate the full c-M200 relation figure (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import m200 as _m200
    return _m200.plot_m200_c_relation_all(*args, **kwargs)


def plot_sample_attrition_pipeline(*args, **kwargs):
    """Generate sample-attrition pipeline figure (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import m200 as _m200
    return _m200.plot_sample_attrition_pipeline(*args, **kwargs)
