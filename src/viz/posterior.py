"""Posterior-distribution visualization (pair plots, ETI/HDI, diagnostics).

Delegates to the original implementations in ``src-orig/dm.py`` and
``src-orig/m200.py`` until the drawing code is fully extracted.
"""


def plot_population_inference_diagnostics(*args, **kwargs):
    """Plot population-model inference diagnostics (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import m200 as _m200
    return _m200.plot_population_inference_diagnostics(*args, **kwargs)


def annotate_pair_marginals(*args, **kwargs):
    """Annotate pair-plot marginal distributions (legacy wrapper)."""
    import sys
    from pathlib import Path as _Path
    _old_root = _Path(__file__).resolve().parent.parent.parent / "src-orig"
    if str(_old_root) not in sys.path:
        sys.path.insert(0, str(_old_root))
    import dm as _dm
    return _dm._annotate_pair_marginals(*args, **kwargs)
